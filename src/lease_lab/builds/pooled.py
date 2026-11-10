"""Offline orchestration: both families develop and freeze before any final generation."""

import json
import time

from lease_lab.artifacts.store import (
    FrozenBundle,
    digest,
    encode,
    publish_bundle,
    source_digest,
    write_json,
)
from lease_lab.builds.exact import build_exact, support_profile
from lease_lab.config import CONFIG, HISTORICAL_OFFERS, PRIOR_SIGMA, ROOT
from lease_lab.models.beta_cells import fit_beta


def build_pooled(activate: bool = True) -> FrozenBundle:
    from lease_lab.evaluation.benchmarks import (
        numerical_checks,
        point_policy,
        policy_scores,
        predictions,
    )
    from lease_lab.evaluation.cohorts import (
        FAMILIES,
        confirmation_rows,
        data_hash,
        development_roles,
        fit_subsets,
        offline_rows,
        primary_ids,
        role_ids,
    )
    from lease_lab.evaluation.coverage import coverage_checks
    from lease_lab.evaluation.metrics import prediction_score
    from lease_lab.evaluation.point_models import DevelopmentFailure, develop_points
    from lease_lab.models.logistic_laplace import LogisticSampler, fit_logistic
    from lease_lab.report_contracts import (
        AuditFinding,
        DatasetIdentity,
        EvaluationReport,
        PointComparison,
        SelectionCheckpoint,
        selection_payload,
    )

    started = time.perf_counter()
    base = build_exact(activate=False)
    provenance = base.provenance.model_copy(
        update={"run_id": "pooled-" + base.provenance.run_id.split("-", 1)[1]}
    )
    directory = ROOT / "artifacts/runs" / provenance.run_id
    config_path = ROOT / "configs/point-comparison-v1.json"
    settings = json.loads(config_path.read_bytes())
    config_hash = digest(config_path.read_bytes())
    if (
        config_hash != CONFIG["point_comparison_config_sha256"]
        or digest((ROOT / "configs/tabpfn-v2.json").read_bytes())
        != settings["tabpfn_config_sha256"]
    ):
        raise ValueError("comparison/pretrained configuration commitment mismatch")
    if (CONFIG["confirmation_seed"], CONFIG["confirmation_namespace"]) != (
        settings["final_seed"],
        settings["final_namespace"],
    ):
        raise ValueError("final recipe differs from committed comparison configuration")
    from lease_lab.simulation.environment import SCENARIO_VERSIONS

    if (
        CONFIG["scenario_versions"] != SCENARIO_VERSIONS
        or settings["scenario_versions"] != SCENARIO_VERSIONS
    ):
        raise ValueError("scenario configuration/version mismatch")
    numerical = numerical_checks()
    prediction, policies, findings, datasets = [], [], [], []
    family_data, bundles, developments, development_ids = {}, {}, {}, {}
    development_started = time.perf_counter()
    deadline = development_started + settings["development_seconds"]

    def save_development_failure(reason, *, family=None, trials=(), fits=0):
        completed = {
            name: {
                "fit_count": result.fit_count,
                "elapsed_seconds": result.elapsed_seconds,
                "methods": [m.model_dump(mode="json") for m in result.records],
            }
            for name, result in developments.items()
        }
        write_json(
            directory / "development-failure.json",
            {
                "provenance": provenance.model_dump(mode="json"),
                "comparison_config_sha256": config_hash,
                "cause": reason,
                "completed_families": completed,
                "failed_family": {
                    "family": family,
                    "fit_count": fits,
                    "trials": [t.model_dump(mode="json") for t in trials],
                }
                if family
                else None,
                "total_fit_attempts": sum(d.fit_count for d in developments.values())
                + fits,
                "development_wall_seconds": time.perf_counter() - development_started,
                "fresh_final_generated": False,
            },
        )

    # Phase 1: only historical fit/development roles. No final contexts or labels exist here.
    for family in FAMILIES:
        rows = offline_rows(family)
        roles = development_roles(rows)
        fit = roles["fit"]
        subsets = fit_subsets(fit, settings["fit_sizes"])
        pooled = {prior: fit_logistic(fit, sd) for prior, sd in PRIOR_SIGMA.items()}
        bundle = base.model_copy(
            update={
                "beta": fit_beta(fit),
                "laplace": pooled,
                "support": support_profile(fit),
            }
        )
        try:
            developed = develop_points(
                fit,
                roles["early_stop"],
                roles["selection"],
                fit_subsets=subsets,
                settings={**settings, "family": family, "deadline": deadline},
            )
        except DevelopmentFailure as error:
            save_development_failure(
                str(error), family=family, trials=error.trials, fits=error.fits
            )
            raise
        if developed.reference.design.base != pooled["default"].transform:
            raise ValueError("point and Bayesian full-fit transforms differ")
        family_data[family], bundles[family], developments[family] = (
            (rows, roles),
            bundle,
            developed,
        )
        development_ids[family] = {
            **{
                k: [e.applicant_id for e in roles[k]]
                for k in ("fit", "diagnostic", "selection")
            },
            "historical_test": [e.applicant_id for e in roles["test"]],
            **role_ids(roles, subsets),
        }
        prediction.extend(
            predictions(
                roles["diagnostic"],
                fit,
                bundle,
                family,
                "diagnostic",
                point_predictions=developed.reference.predict(roles["diagnostic"]),
            )
        )
        for name, model in developed.models.items():
            prediction.append(
                prediction_score(
                    roles["report_only"],
                    model.predict(roles["report_only"]),
                    family,
                    name,
                    "report-only diagnostic",
                    "not_estimated",
                    bundle=bundle,
                )
            )
        for prior, state in pooled.items():
            findings.append(
                AuditFinding(
                    name=f"{family} · {prior} prior · MAP diagnostics",
                    status="converged" if state.diagnostics.converged else "fallback",
                    detail=state.diagnostics.model_dump_json(),
                )
            )
    development_seconds = sum(d.elapsed_seconds for d in developments.values())
    fit_count = sum(d.fit_count for d in developments.values())
    if fit_count > settings["maximum_added_fits"] or time.perf_counter() > deadline:
        save_development_failure(
            "development budget exceeded; final reserve remains unopened"
        )
        raise ValueError("development budget exceeded; final reserve remains unopened")
    comparison = PointComparison(
        version=settings["version"],
        scenario_versions=settings["scenario_versions"],
        config_sha256=config_hash,
        authority_version=settings["authority_version"],
        authority_sha256=settings["authority_sha256"],
        final_seed=settings["final_seed"],
        final_namespace=settings["final_namespace"],
        freeze_sha256="0" * 64,
        status="awaiting_tabpfn",
        development_fit_count=fit_count,
        development_seconds=development_seconds,
        baseline_fit_count=0,
        methods=[m for d in developments.values() for m in d.records],
    )
    checkpoint = SelectionCheckpoint(
        provenance=provenance,
        comparison=comparison,
        roles=development_ids,
        data_sha256={
            family: data_hash(rows) for family, (rows, _) in family_data.items()
        },
        frozen_before_final=True,
        pre_generation_run_id=provenance.run_id,
        pre_generation_sha256=None,
    )
    comparison = comparison.model_copy(
        update={"freeze_sha256": digest(encode(selection_payload(checkpoint)))}
    )
    checkpoint = checkpoint.model_copy(update={"comparison": comparison})
    if source_digest() != provenance.source_sha256:
        raise ValueError("source changed during development; do not open final reserve")
    write_json(directory / "selection.json", checkpoint)
    pre_generation_hash = digest((directory / "selection.json").read_bytes())
    # Phase 2: both family decisions and pretrained context/config are now durably committed.
    for family in FAMILIES:
        rows, roles = family_data[family]
        fit, bundle, developed = roles["fit"], bundles[family], developments[family]
        final = confirmation_rows(family)
        split_ids = primary_ids(rows, final)
        datasets.append(
            DatasetIdentity(
                family=family,
                data_sha256=data_hash(rows),
                split_ids_sha256=digest(encode(split_ids)),
                applicants=len(rows),
                confirmation_seed=settings["final_seed"],
                confirmation_namespace=settings["final_namespace"],
                confirmation_data_sha256=data_hash(final),
                **{s: len(ids) for s, ids in split_ids.items()},
            )
        )
        prediction.extend(
            predictions(
                final,
                fit,
                bundle,
                family,
                "sealed test",
                point_predictions=developed.reference.predict(final),
            )
        )
        policies.extend(policy_scores(final, bundle, family))
        for prior, state in bundle.laplace.items():
            p = LogisticSampler(state).predict(
                [e.context for e in final], [e.offer_id for e in final]
            )[0]
            prediction.append(
                prediction_score(
                    final,
                    p,
                    family,
                    "laplace-prior-" + prior,
                    "sealed test",
                    "map-laplace-approximation",
                    bundle=bundle,
                )
            )
        for name, model in developed.models.items():
            before = time.perf_counter()
            prediction.append(
                prediction_score(
                    final,
                    model.predict(final),
                    family,
                    name,
                    "sealed test",
                    "not_estimated",
                    bundle=bundle,
                )
            )
            policies.append(
                point_policy(
                    final,
                    model.all_offers(final, HISTORICAL_OFFERS),
                    bundle,
                    family,
                    name,
                    offer_ids=HISTORICAL_OFFERS,
                )
            )
            record = next(
                m for m in comparison.methods if m.family == family and m.model == name
            )
            record.prediction_seconds = time.perf_counter() - before
        if family == "smooth":
            base = bundle.model_copy(update={"provenance": provenance})
            write_json(
                directory / "split-ids.json",
                {
                    "provenance": provenance.model_dump(mode="json"),
                    "ids": split_ids,
                    "development_ids": role_ids(
                        roles, fit_subsets(fit, settings["fit_sizes"])
                    ),
                },
            )
    coverage = coverage_checks()
    base = base.model_copy(
        update={
            "provenance": provenance,
            "stages": base.stages
            + [
                "2: MAP/Laplace, independent references, held-out and repeated-data coverage",
                "5: bounded logistic/GBDT development; committed pretrained TabPFN comparison",
            ],
        }
    )
    report = EvaluationReport(
        provenance=provenance,
        datasets=datasets,
        comparison=comparison,
        prediction=prediction,
        numerical=numerical,
        coverage=coverage,
        policies=policies,
        findings=findings,
        commands=["uv run lease-lab build --through pooled", "uv run pytest"],
        elapsed_seconds=time.perf_counter() - started,
        interpretation=[
            "Every behavioural/economic result is simulation-relative; no real commercial claim.",
            "30 fixed-world repeated-data datasets per probe; Wilson intervals reflect finite seed count, not nominal coverage guarantees.",
            "Beta targets cohort-average probability; Laplace targets full visible-context probability. This is not simulation-based calibration.",
            "Both families completed development and saved selection.json before generating any fresh final rows. Evaluator probabilities never fit or choose a model.",
            f"Primary sealed-test scores use 900 fresh applicants per family: seed {settings['final_seed']}, namespace {settings['final_namespace']}. Historical confirmation remains repair evidence.",
            "Same-512 controls share fitting IDs/base float32 matrices, but full-fit preprocessing contexts/support history (3600) and external TabPFN pretraining are not equalized.",
            "Selected logistic consumes 4200 distinct fit/selection labels and monitors 450 others; selected GBDT consumes 4650 including the same 450 stopping/monitoring labels. Calibration labels=0.",
            "Selected methods have report-only diagnostic N=450; original Bayesian diagnostics retain all 900 previously exposed diagnostic rows.",
            "Exact Beta, empirical, original logistic and Laplace references use 3600 fit labels with fixed settings; no selection/calibration labels choose their settings.",
            "Two fixed full-fit logistic reference configurations are reused from base C=1 search fits, with zero extra reference fits. The separate response-change Platt task uses 900 calibration labels.",
            "Development curves condition on selected settings and shared full-fit preprocessing; they are not fresh generalization estimates.",
            "Policy-value intervals are paired applicant bootstraps after freeze, not posterior intervals. Point methods cannot enter P5 or Thompson.",
            "All applicants, including manual reviews at -€15, remain in the denominator.",
            "Advisory support is a separate range/count heuristic, not confidence or a calibrated out-of-distribution detector.",
            "True TabPFN fine-tuning is deferred; this comparison requires the unchanged pretrained V2 benchmark before completion.",
        ],
    )
    publish_bundle(base, activate=False)
    write_json(directory / "evaluation.json", report)
    write_json(
        directory / "manifest.json",
        {
            "provenance": provenance.model_dump(mode="json"),
            "command": "uv run lease-lab build --through pooled",
            "elapsed_seconds": report.elapsed_seconds,
            "settings": CONFIG,
            "comparison_config_sha256": config_hash,
            "selection_freeze_sha256": comparison.freeze_sha256,
            "pre_generation_checkpoint_sha256": pre_generation_hash,
            "development_fit_count": fit_count,
            "development_seconds": development_seconds,
            "extra_reference_fits": 0,
            "reused_reference_configurations": 2,
            "python_lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
            "frontend_lock_sha256": digest(
                (ROOT / "ui/package-lock.json").read_bytes()
            ),
        },
    )
    if activate:
        publish_bundle(base)
    return base
