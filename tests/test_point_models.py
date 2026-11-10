"""Small non-final fixtures for fitting, public curves, roles and identity boundaries."""

import copy
import json
import time

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from lease_lab.artifacts import store
from lease_lab.artifacts.store import FrozenBundle, digest, encode
from lease_lab.config import CONFIG, HISTORICAL_OFFERS, ROOT
from lease_lab.contracts import DecisionRequest
from lease_lab.data.features import fit_transform
from lease_lab.evaluation import benchmarks, cohorts, coverage
from lease_lab.evaluation.point_models import (
    DevelopmentFailure,
    choose_trial,
    develop_points,
    design_for,
    native_stop,
)
from lease_lab.report_contracts import (
    EvaluationReport,
    PointComparison,
    PolicyScore,
    SelectionCheckpoint,
    require_completed_comparison,
    selection_payload,
)
from lease_lab.simulation.environment import generate


@pytest.fixture(scope="module")
def fixture():
    rows = generate(200, 47)
    roles = cohorts.development_roles(rows)
    settings = json.loads((ROOT / "configs/point-comparison-v1.json").read_bytes())
    settings["fit_sizes"] = [32, 64, 120]
    settings["gbdt"]["max_iter"] = 25
    settings["fixed_gbdt"]["max_iter"] = 8
    settings.update(family="smooth", deadline=time.perf_counter() + 30)
    subsets = cohorts.fit_subsets(roles["fit"], settings["fit_sizes"])
    developed = develop_points(
        roles["fit"],
        roles["early_stop"],
        roles["selection"],
        fit_subsets=subsets,
        settings=settings,
    )
    raw = json.loads(
        (ROOT / "artifacts/runs/core-01e96c4ee2b1/bundle.json").read_bytes()
    )
    raw["provenance"]["schema_version"] = "1.2"
    return roles, settings, subsets, developed, FrozenBundle.model_validate(raw)


def test_roles_fit_only_design_and_identical_controls(fixture):
    roles, settings, subsets, developed, _ = fixture
    assert set(e.applicant_id for e in roles["early_stop"]).isdisjoint(
        e.applicant_id for e in roles["report_only"]
    )
    assert sorted(
        e.applicant_id for e in roles["early_stop"] + roles["report_only"]
    ) == sorted(e.applicant_id for e in roles["diagnostic"])
    assert [e.applicant_id for e in roles["early_stop"]] == [
        e.applicant_id for e in cohorts.stable_rows(roles["diagnostic"])[:15]
    ]
    assert developed.fit_count == 21
    selected = [m for m in developed.records if m.selection == "selected"]
    assert [len(m.trials) for m in selected] == [9, 5]
    base = fit_transform(roles["fit"])
    fixed = [m for m in developed.records if m.selection != "selected"]
    assert len({m.budget.fit.ids_sha256 for m in fixed}) == 1
    assert len({m.fit_matrix_sha256 for m in fixed}) == 1
    assert len({m.fit_data_sha256 for m in fixed}) == 1
    np.testing.assert_array_equal(
        developed.models["gbdt-point-512"].design.matrix(subsets[32]),
        base.matrix(
            [e.context for e in subsets[32]], [e.offer_id for e in subsets[32]]
        ).astype(np.float32),
    )
    interacting = design_for(roles["fit"], base, "rate-interactions")
    X = base.matrix(
        [e.context for e in roles["fit"]], [e.offer_id for e in roles["fit"]]
    )
    np.testing.assert_allclose(
        interacting.identity.derived_mean,
        (X[:, 4, None] * X[:, [3, 5, 6, 7]]).mean(axis=0),
    )
    event = roles["selection"][0]
    alternatives = interacting.matrix([event] * 3, list(HISTORICAL_OFFERS))
    assert len(set(alternatives[:, 4])) == 3
    assert len(set(alternatives[:, 3])) == 1
    assert not np.array_equal(alternatives[0, 8:], alternatives[2, 8:])
    for method in selected:
        assert [c.fit_n for c in method.curves] == [32, 64, 120]
        assert method.budget.calibration.n == 0
        assert method.feature.base_mean == base.mean
    poisoned = [
        e.model_copy(update={"accepted": not e.accepted}) for e in roles["test"]
    ]
    settings = {**settings, "deadline": time.perf_counter() + 30}
    again = develop_points(
        roles["fit"],
        roles["early_stop"],
        roles["selection"],
        fit_subsets=subsets,
        settings=settings,
    )
    assert [(m.chosen_candidate, m.parameters, m.feature) for m in again.records] == [
        (m.chosen_candidate, m.parameters, m.feature) for m in developed.records
    ]
    with pytest.raises(ValueError, match="wrong role"):
        develop_points(
            roles["fit"] + poisoned,
            roles["early_stop"],
            roles["selection"],
            fit_subsets=subsets,
            settings=settings,
        )
    with pytest.raises(ValueError, match="distinct"):
        develop_points(
            roles["fit"],
            roles["early_stop"] + roles["early_stop"],
            roles["selection"],
            fit_subsets=subsets,
            settings=settings,
        )


def test_native_tree_round_zero_curve_prefix_and_stop(fixture):
    roles, settings, _, _, _ = fixture
    transform = fit_transform(roles["fit"])
    design = design_for(roles["fit"], transform, "base")
    X, V = design.matrix(roles["fit"]), design.matrix(roles["early_stop"])
    y, v = [e.accepted for e in roles["fit"]], [e.accepted for e in roles["early_stop"]]
    parameters = dict(
        learning_rate=0.1,
        max_leaf_nodes=7,
        l2_regularization=1.0,
        min_samples_leaf=20,
        max_iter=12,
        random_state=47,
        n_iter_no_change=5,
        tol=1e-4,
        categorical_features=None,
    )
    tree = HistGradientBoostingClassifier(
        **parameters, early_stopping=True, scoring="loss"
    ).fit(X, y, X_val=V, y_val=v)
    assert len(tree.train_score_) == tree.n_iter_ + 1
    assert -tree.train_score_[0] == pytest.approx(
        log_loss(y, np.full(len(y), np.mean(y)))
    )
    for i, (train_p, val_p) in enumerate(
        zip(tree.staged_predict_proba(X), tree.staged_predict_proba(V)), 1
    ):
        assert -tree.train_score_[i] == pytest.approx(log_loss(y, train_p), abs=1e-13)
        assert -tree.validation_score_[i] == pytest.approx(
            log_loss(v, val_p), abs=1e-13
        )
    chosen = int(np.argmin(-tree.validation_score_[1:])) + 1
    refit = HistGradientBoostingClassifier(
        **(parameters | {"max_iter": chosen}), early_stopping=False
    ).fit(X, y)
    for matrix in (X, V):
        expected = next(
            p for i, p in enumerate(tree.staged_predict_proba(matrix), 1) if i == chosen
        )
        np.testing.assert_allclose(
            refit.predict_proba(matrix), expected, atol=1e-10, rtol=0
        )
    assert native_stop([0.7, 0.6, 0.61, 0.62], 2, 1e-4)
    assert not native_stop([0.7, 0.6, 0.59, 0.58], 2, 1e-4)
    # Cap and stop are independent facts, including a coincident final round.
    assert tree.n_iter_ == 12 or native_stop(
        (-tree.validation_score_).tolist(), 5, 1e-4
    )


def test_ties_failures_and_complete_budget_accounting(fixture):
    roles, settings, subsets, developed, _ = fixture
    trials = developed.records[0].trials
    a = trials[0].model_copy(update={"selection_log_loss": 0.4})
    b = trials[1].model_copy(update={"selection_log_loss": 0.4 + 5e-9})
    assert choose_trial([b, a], "logistic", 1e-8).candidate_id == a.candidate_id
    failed = a.model_copy(
        update={
            "status": "failed",
            "failure": "observed fixture failure",
            "selection_log_loss": None,
        }
    )
    assert choose_trial([failed, b], "logistic", 1e-8) == b
    with pytest.raises(ValueError, match="all candidates"):
        choose_trial([failed], "logistic", 1e-8)
    with pytest.raises(DevelopmentFailure) as error:
        develop_points(
            roles["fit"],
            roles["early_stop"],
            roles["selection"],
            fit_subsets=subsets,
            settings=settings | {"deadline": 0},
        )
    assert len(error.value.trials) == 14 and error.value.fits == 0
    assert all(
        t.status == "budget_cutoff" and t.selection_log_loss is None
        for t in error.value.trials
    )
    nonconverged = copy.deepcopy(settings)
    nonconverged["logistic"]["max_iter"] = 1
    nonconverged["deadline"] = time.perf_counter() + 30
    with pytest.raises(DevelopmentFailure) as error:
        develop_points(
            roles["fit"],
            roles["early_stop"],
            roles["selection"],
            fit_subsets=subsets,
            settings=nonconverged,
        )
    assert all(
        t.status == "failed" and "converge" in t.failure for t in error.value.trials[:9]
    )


def test_point_policy_whole_matrix_guard_and_original_fixture(fixture):
    roles, _, _, _, bundle = fixture
    rows = roles["selection"][:6]
    golden = json.loads(
        (ROOT / "tests/fixtures/original-point-policy.json").read_bytes()
    )
    p = np.asarray(golden["probabilities"])
    expected = PolicyScore.model_validate(golden["expected"])
    actual = benchmarks.point_policy(
        rows, p, bundle, "smooth", "tabpfn-v2-512", offer_ids=HISTORICAL_OFFERS
    )
    assert actual == expected
    for invalid in (
        p[:2],
        p[:, 0],
        np.zeros((6, 2)),
        np.full((6, 3), np.nan),
        np.full((6, 3), 1.1),
        np.full((6, 3), -0.1),
    ):
        with pytest.raises(ValueError):
            benchmarks.point_policy(
                rows, invalid, bundle, "smooth", "point", offer_ids=HISTORICAL_OFFERS
            )
    with pytest.raises(ValueError):
        benchmarks.point_policy(
            rows,
            p,
            bundle,
            "smooth",
            "point",
            offer_ids=tuple(reversed(HISTORICAL_OFFERS)),
        )
    rare = rows[0].model_copy(
        update={
            "context": rows[0].context.model_copy(
                update={"segment": "restart", "existing_debt_burden": 1.8}
            )
        }
    )
    review = benchmarks.point_policy(
        [rare],
        np.full((1, 3), 0.9),
        bundle,
        "smooth",
        "point",
        offer_ids=HISTORICAL_OFFERS,
    )
    assert (
        review.n == review.reviews == 1
        and review.mean_expected_contribution_cents == -1500
    )
    new_card = rows[0].model_copy(
        update={
            "context": next(c for c in bundle.case_presets if c.segment == "growing"),
            "card_id": "dec-2026",
        }
    )
    review = benchmarks.point_policy(
        [new_card],
        np.full((1, 3), 0.9),
        bundle,
        "smooth",
        "point",
        offer_ids=HISTORICAL_OFFERS,
    )
    assert review.n == review.reviews == 1 and review.support_violations == 0
    assert review.mean_expected_contribution_cents == -1500
    for method in ("gbdt-point-selected", "logistic-point-512", "tabpfn-v2-512"):
        with pytest.raises(ValidationError):
            DecisionRequest(method=method)


@pytest.mark.parametrize("failure", [None, "nonlinear", "budget"])
def test_both_families_checkpoint_exists_before_final_generation(
    tmp_path, monkeypatch, fixture, failure
):
    from lease_lab.builds import pooled
    from lease_lab.models import logistic_laplace

    roles, settings, _, developed, bundle = fixture
    settings = {k: v for k, v in settings.items() if k not in ("family", "deadline")}
    if failure == "budget":
        settings["maximum_added_fits"] = 41
    configs = tmp_path / "configs"
    configs.mkdir()
    for filename in ("point-comparison-v1.json", "tabpfn-v2.json"):
        value = (
            encode(settings)
            if filename.startswith("point")
            else (ROOT / "configs" / filename).read_bytes()
        )
        (configs / filename).write_bytes(value)
    monkeypatch.setitem(
        CONFIG,
        "point_comparison_config_sha256",
        digest((configs / "point-comparison-v1.json").read_bytes()),
    )
    monkeypatch.setattr(pooled, "ROOT", tmp_path)
    monkeypatch.setattr(pooled, "build_exact", lambda **kwargs: bundle)
    monkeypatch.setattr(pooled, "support_profile", lambda fit: bundle.support)
    monkeypatch.setattr(
        pooled, "source_digest", lambda: bundle.provenance.source_sha256
    )
    monkeypatch.setattr(
        cohorts, "offline_rows", lambda family: generate(200, 47, "smooth")
    )
    monkeypatch.setattr(benchmarks, "numerical_checks", lambda: [])
    monkeypatch.setattr(coverage, "coverage_checks", lambda: [])
    monkeypatch.setattr(benchmarks, "predictions", lambda *args, **kwargs: [])
    transform = fit_transform(roles["fit"])
    monkeypatch.setattr(
        logistic_laplace,
        "fit_logistic",
        lambda fit, sd: bundle.laplace["default"].model_copy(
            update={"transform": transform}
        ),
    )
    seen = []
    from lease_lab.evaluation import point_models

    def develop(*args, **kwargs):
        seen.append(kwargs["settings"]["family"])
        if failure == "nonlinear" and seen[-1] == "nonlinear":
            raise DevelopmentFailure(
                "injected second-family failure", developed.records[0].trials[:2], 2
            )
        return point_models.PointDevelopment(
            developed.models,
            [m.model_copy(update={"family": seen[-1]}) for m in developed.records],
            developed.reference,
            21,
            0.1,
        )

    monkeypatch.setattr(point_models, "develop_points", develop)

    class FinalBoundaryReached(Exception):
        pass

    def final(family):
        assert seen == ["smooth", "nonlinear"]
        paths = list((tmp_path / "artifacts/runs").glob("*/selection.json"))
        assert len(paths) == 1
        checkpoint = SelectionCheckpoint.model_validate_json(paths[0].read_bytes())
        assert {m.family for m in checkpoint.comparison.methods} == {
            "smooth",
            "nonlinear",
        }
        assert (
            digest(encode(selection_payload(checkpoint)))
            == checkpoint.comparison.freeze_sha256
        )
        raise FinalBoundaryReached

    monkeypatch.setattr(cohorts, "confirmation_rows", final)
    expected_error = (
        FinalBoundaryReached
        if failure is None
        else DevelopmentFailure
        if failure == "nonlinear"
        else ValueError
    )
    with pytest.raises(expected_error):
        pooled.build_pooled(activate=False)
    if failure is not None:
        assert seen == ["smooth", "nonlinear"]
        assert not list((tmp_path / "artifacts/runs").glob("*/selection.json"))
        failures = list(
            (tmp_path / "artifacts/runs").glob("*/development-failure.json")
        )
        assert len(failures) == 1
        evidence = json.loads(failures[0].read_bytes())
        completed = evidence["completed_families"]
        assert set(completed) == (
            {"smooth"} if failure == "nonlinear" else {"smooth", "nonlinear"}
        )
        for family, saved in completed.items():
            assert saved["fit_count"] == 21
            assert saved["elapsed_seconds"] == 0.1
            assert saved["methods"] == [
                m.model_copy(update={"family": family}).model_dump(mode="json")
                for m in developed.records
            ]
        if failure == "nonlinear":
            assert evidence["failed_family"] == {
                "family": "nonlinear",
                "fit_count": 2,
                "trials": [
                    t.model_dump(mode="json") for t in developed.records[0].trials[:2]
                ],
            }
        else:
            assert evidence["failed_family"] is None
        assert evidence["total_fit_attempts"] == (23 if failure == "nonlinear" else 42)
        assert evidence["fresh_final_generated"] is False
        assert evidence["development_wall_seconds"] >= 0
        assert (
            evidence["provenance"]["source_sha256"] == bundle.provenance.source_sha256
        )
        assert (
            evidence["comparison_config_sha256"]
            == CONFIG["point_comparison_config_sha256"]
        )


def test_final_inventory_and_sidecar_identity_rejections(fixture):
    _, settings, _, developed, bundle = fixture
    repair = EvaluationReport.model_validate_json(
        (ROOT / "artifacts/runs/core-d1e450cb9856/evaluation.json").read_bytes()
    )
    methods = []
    for family in ("smooth", "nonlinear"):
        for original in developed.records:
            method = original.model_copy(deep=True)
            method.family, method.status = family, "completed"
            b = method.budget
            selected = method.selection == "selected"
            b.fit.n, b.selection.n, b.early_stop.n = (
                (3600 if selected else 512),
                (600 if selected else 0),
                (450 if method.model == "gbdt-point-selected" else 0),
            )
            b.monitoring.n = 450 if selected else 0
            b.consumed_union_n = b.fit.n + b.selection.n + b.early_stop.n
            b.observed_union_n = b.fit.n + b.selection.n + b.monitoring.n
            b.preprocessing_context_n = b.support_history_n = 3600
            for curve, size in zip(method.curves, [512, 1800, 3600]):
                curve.fit_n = size
            methods.append(method)
    comparison = PointComparison(
        version="point-comparison-v1",
        scenario_versions=settings["scenario_versions"],
        config_sha256="a" * 64,
        authority_version="2.2",
        authority_sha256=settings["authority_sha256"],
        final_seed=20261111,
        final_namespace="comparison-confirmation-v1",
        freeze_sha256="b" * 64,
        status="completed",
        development_fit_count=42,
        development_seconds=0.2,
        baseline_fit_count=0,
        methods=methods,
    )
    prediction, policies = repair.prediction.copy(), repair.policies.copy()
    for method in methods:
        prediction.append(
            next(
                s
                for s in repair.prediction
                if s.family == method.family and s.cohort == "sealed test"
            ).model_copy(
                update={"model": method.model, "uncertainty_method": "not_estimated"}
            )
        )
        policies.append(
            next(
                s
                for s in repair.policies
                if s.family == method.family and s.policy == "greedy"
            ).model_copy(update={"model": method.model})
        )
    report = EvaluationReport.model_validate(
        repair.model_dump()
        | {
            "comparison": comparison,
            "prediction": prediction,
            "policies": policies,
            "datasets": [
                d.model_copy(
                    update={
                        "confirmation_seed": 20261111,
                        "confirmation_namespace": "comparison-confirmation-v1",
                    }
                )
                for d in repair.datasets
            ],
        }
    )
    require_completed_comparison(report)
    for patch in (
        {"comparison": None},
        {"prediction": prediction[:-1]},
        {"policies": policies[:-1]},
        {"comparison": comparison.model_copy(update={"status": "awaiting_tabpfn"})},
    ):
        with pytest.raises(ValueError):
            require_completed_comparison(report.model_copy(update=patch))
    with pytest.raises(ValidationError):
        EvaluationReport.model_validate(
            report.model_dump() | {"prediction": prediction + [prediction[-1]]}
        )
    # Minimal source-matched sidecar fixture uses historical rows only; no final generation.
    from lease_lab.artifacts.challenger import ChallengerReport

    raw = json.loads(
        (ROOT / "artifacts/challengers/core-01e96c4ee2b1/tabpfn-v2.json").read_bytes()
    )
    raw.update(
        provenance=bundle.provenance.model_dump(mode="json"),
        freeze_sha256="b" * 64,
        comparison_config_sha256="a" * 64,
        prediction=[
            p.model_dump(mode="json") for p in prediction if p.model == "tabpfn-v2-512"
        ],
        policies=[
            p.model_dump(mode="json") for p in policies if p.model == "tabpfn-v2-512"
        ],
    )
    for item in raw["inputs"]:
        method = next(
            m
            for m in methods
            if m.family == item["family"] and m.model == "tabpfn-v2-512"
        )
        method.budget.fit.ids_sha256 = digest(encode(item["context_ids"]))
        item.update(
            freeze_sha256="b" * 64,
            comparison_config_sha256="a" * 64,
            feature_sha256=method.feature.transform_sha256,
            context_matrix_sha256=method.fit_matrix_sha256,
            context_sha256=method.fit_data_sha256,
        )
    extra = ChallengerReport.model_validate(raw)
    store.validate_challenger(extra, report)
    for field in (
        "freeze_sha256",
        "comparison_config_sha256",
        "confirmation_sha256",
        "context_matrix_sha256",
        "feature_sha256",
        "context_sha256",
    ):
        changed = extra.model_copy(deep=True)
        setattr(changed.inputs[0], field, "c" * 64)
        with pytest.raises(ValueError):
            store.validate_challenger(changed, report)
    with pytest.raises(ValidationError):
        ChallengerReport.model_validate(
            extra.model_dump()
            | {"prediction": extra.prediction + [extra.prediction[0]]}
        )
