"""Real offline TabPFN V2 point prediction; no posterior or Thompson adapter."""

import importlib.metadata
import time

import numpy as np

from lease_lab.artifacts.challenger import ChallengerInput, ChallengerReport
from lease_lab.artifacts.store import (
    digest,
    encode,
    load_bundle,
    load_report,
    source_digest,
    write_json,
)
from lease_lab.config import HISTORICAL_OFFERS, ROOT
from lease_lab.evaluation.metrics import prediction_score
from lease_lab.evaluation.benchmarks import point_policy
from lease_lab.evaluation.cohorts import confirmation_rows, fit_subsets, offline_rows
from lease_lab.evaluation.point_models import design_for
from lease_lab.evaluation.tabpfn_runtime import (
    checkpoint,
    configuration,
    offline_inference,
    peak_memory_bytes,
    prepare_environment,
)
from lease_lab.models.beta_cells import fit_beta
from lease_lab.builds.exact import support_profile
from lease_lab.data.features import Transform, fit_transform
from lease_lab.report_contracts import AuditFinding


def run_challenger() -> ChallengerReport:
    started = time.perf_counter()
    config = configuration()
    bundle = load_bundle()
    if bundle.provenance.source_sha256 != source_digest():
        raise ValueError(
            "reproduce the core with current code/config before the challenger"
        )
    base = load_report(bundle, include_challenger=False)
    if base.comparison is None or base.comparison.status != "awaiting_tabpfn":
        raise ValueError("frozen core comparison required before pretrained inference")
    comparison = base.comparison
    if (
        digest((ROOT / "configs/tabpfn-v2.json").read_bytes())
        != next(m for m in comparison.methods if m.model == "tabpfn-v2-512").parameters[
            "configuration_sha256"
        ]
    ):
        raise ValueError("pretrained configuration changed after selection freeze")
    model_path = checkpoint(config)
    prepare_environment()
    scores = []
    policies = []
    inputs = []
    with offline_inference(config["max_seconds"]):
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if importlib.metadata.version("tabpfn") != config["library_version"]:
            raise ValueError("unexpected TabPFN version")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        for family in ("smooth", "nonlinear"):
            rows = offline_rows(family)
            full_fit = [e for e in rows if e.split == "fit"]
            fit = fit_subsets(full_fit)[config["context_applicants"]]
            committed = next(
                m
                for m in comparison.methods
                if m.family == family and m.model == "tabpfn-v2-512"
            )
            transform = Transform(
                mean=committed.feature.base_mean, scale=committed.feature.base_scale
            )
            if transform != fit_transform(full_fit):
                raise ValueError(
                    "family fit transform differs from pre-generation commitment"
                )
            family_bundle = bundle.model_copy(
                update={
                    "beta": fit_beta(full_fit),
                    "support": support_profile(full_fit),
                }
            )
            test = confirmation_rows(family)
            assert not ({e.applicant_id for e in fit} & {e.applicant_id for e in test})
            identity = next(d for d in base.datasets if d.family == family)
            assert (
                digest(encode([e.model_dump(mode="json") for e in test]))
                == identity.confirmation_data_sha256
            )
            X = transform.matrix(
                [e.context for e in fit], [e.offer_id for e in fit]
            ).astype(np.float32)
            y = np.array([e.accepted for e in fit], dtype=int)
            model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                model_path=model_path,
                device="cpu",
                n_estimators=config["estimators"],
                n_preprocessing_jobs=1,
                random_state=config["seed"],
                fit_mode="fit_preprocessors",
                inference_precision=torch.float32,
                enable_torch_compile=False,
            )
            fitted = time.perf_counter()
            model.fit(X, y)
            fit_seconds = time.perf_counter() - fitted
            if model.n_estimators_ != config["estimators"]:
                raise ValueError("unexpected estimator expansion")

            def predict(matrix):
                positive = int(np.flatnonzero(model.classes_ == 1)[0])
                chunks = [
                    model.predict_proba(matrix[i : i + config["query_chunk_size"]])[
                        :, positive
                    ]
                    for i in range(0, len(matrix), config["query_chunk_size"])
                ]
                values = np.concatenate(chunks)
                if peak_memory_bytes() > config["memory_target_bytes"]:
                    raise MemoryError("challenger exceeded declared memory target")
                return values

            Xtest = transform.matrix(
                [e.context for e in test], [e.offer_id for e in test]
            ).astype(np.float32)
            predicted = time.perf_counter()
            p = predict(Xtest)
            repeated = predict(Xtest[: config["query_chunk_size"]])
            np.testing.assert_allclose(
                p[: len(repeated)], repeated, atol=1e-6, rtol=1e-6
            )
            scores.append(
                prediction_score(
                    test,
                    p,
                    family,
                    "tabpfn-v2-512",
                    "sealed test",
                    "not_estimated",
                    bundle=family_bundle,
                )
            )
            counter_contexts = [e.context for e in test for _ in HISTORICAL_OFFERS]
            counter_offers = list(HISTORICAL_OFFERS) * len(test)
            counter = predict(
                transform.matrix(counter_contexts, counter_offers).astype(np.float32)
            ).reshape(len(test), 3)
            policies.append(
                point_policy(
                    test,
                    counter,
                    family_bundle,
                    family,
                    "tabpfn-v2-512",
                    offer_ids=HISTORICAL_OFFERS,
                )
            )
            inputs.append(
                ChallengerInput(
                    family=family,
                    freeze_sha256=comparison.freeze_sha256,
                    comparison_config_sha256=comparison.config_sha256,
                    feature_sha256=design_for(
                        full_fit, transform, "base", "float32"
                    ).identity.transform_sha256,
                    context_matrix_sha256=digest(X.tobytes()),
                    context_ids=[e.applicant_id for e in fit],
                    context_sha256=digest(
                        encode([e.model_dump(mode="json") for e in fit])
                    ),
                    confirmation_sha256=identity.confirmation_data_sha256,
                    fit_seconds=fit_seconds,
                    prediction_seconds=time.perf_counter() - predicted,
                )
            )
            print(
                f"{family}: real TabPFN Brier {scores[-1].brier:.5f}; mean policy contribution EUR{policies[-1].mean_expected_contribution_cents / 100:.2f}",
                flush=True,
            )
        result = ChallengerReport(
            provenance=bundle.provenance,
            freeze_sha256=comparison.freeze_sha256,
            comparison_config_sha256=comparison.config_sha256,
            kind="tabular-transformer-point-benchmark",
            library_version=config["library_version"],
            torch_version=torch.__version__,
            checkpoint_repo=config["checkpoint_repo"],
            checkpoint_revision=config["checkpoint_revision"],
            checkpoint_filename=config["checkpoint_filename"],
            checkpoint_sha256=config["checkpoint_sha256"],
            license_url=config["license_url"],
            device="cpu",
            context_applicants=config["context_applicants"],
            estimators=config["estimators"],
            uncertainty_method="not_estimated",
            inputs=inputs,
            prediction=scores,
            policies=policies,
            findings=[
                AuditFinding(
                    name="Tabular transformer challenger",
                    status="point-only",
                    detail=f"Real TabPFN {config['library_version']}, older V2 checkpoint {config['checkpoint_filename']} at {config['checkpoint_revision']}, SHA256 {config['checkpoint_sha256']}. 512 whole fit applicants selected by stable ID hash, 1 estimator, CPU float32, no calibration. Same 900 confirmation applicants/visible feature design; fixed logistic and GBDT controls use the same 512 labels/base float32 matrix; all share full-fit preprocessing/support history. External pretraining is unequalized. Selected models have additional development labels. Inference network access disabled. Epistemic uncertainty not estimated; no P5 or Thompson use. Built with PriorLabs-TabPFN. Licence: {config['license_url']}.",
                )
            ],
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_bytes=peak_memory_bytes(),
        )
    write_json(
        ROOT / "artifacts/challengers" / bundle.provenance.run_id / "tabpfn-v2.json",
        result,
    )
    return result
