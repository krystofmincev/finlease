"""Private replay failures and prediction-denominator semantics; no real path faults."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from lease_lab.artifacts import store
from lease_lab.artifacts.store import FrozenBundle, Current, digest, encode, write_json
from lease_lab.config import ROOT
from lease_lab.contracts import Context, Finance
from lease_lab.decisions.support import support_for
from lease_lab.evaluation.metrics import losses, prediction_score
from lease_lab.models.beta_cells import fit_beta
from lease_lab.report_contracts import PredictionScore, ReplayReport
from lease_lab.simulation.environment import generate


@pytest.fixture
def bundle():
    raw = json.loads(
        (ROOT / "artifacts/runs/core-01e96c4ee2b1/bundle.json").read_bytes()
    )
    raw["provenance"]["schema_version"] = "1.2"
    return FrozenBundle.model_validate(raw)


@pytest.mark.parametrize("fault", ["temp_write", "replace", "encode"])
def test_atomic_replay_failure_and_compact_retry(tmp_path, monkeypatch, bundle, fault):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    directory = tmp_path / "artifacts/runs" / bundle.provenance.run_id
    path = directory / "replays.json"
    old = ReplayReport(
        provenance=bundle.provenance,
        paths=[],
        summaries=[],
        scenarios=[],
        seed_ids=[],
        initial_evidence_count=0,
        decisions_per_seed=0,
        elapsed_seconds=0.0,
        interpretation=["old valid replay"],
    )
    write_json(directory / "bundle.json", bundle)
    pointer = tmp_path / "artifacts/current.json"
    write_json(
        pointer,
        Current(run_id=bundle.provenance.run_id, bundle_sha256=digest(encode(bundle))),
    )
    write_json(path, old, compact=True)
    before, selected = path.read_bytes(), pointer.read_bytes()
    updated = old.model_copy(update={"interpretation": ["successful retry"]})
    with monkeypatch.context() as patch:
        if fault == "temp_write":
            original = Path.write_bytes

            def interrupted(target, data):
                if target == path.with_suffix(".json.tmp"):
                    original(target, data[:17])
                    raise OSError("injected prefix write")
                return original(target, data)

            patch.setattr(Path, "write_bytes", interrupted)
        elif fault == "replace":
            original = Path.replace

            def interrupted(target, destination):
                if target == path.with_suffix(".json.tmp"):
                    raise OSError("injected replace failure")
                return original(target, destination)

            patch.setattr(Path, "replace", interrupted)
        value = {"invalid": float("nan")} if fault == "encode" else updated
        with pytest.raises((OSError, ValueError)):
            write_json(path, value, compact=True)
    assert path.read_bytes() == before and pointer.read_bytes() == selected
    assert store.load_replays(bundle) == old
    write_json(path, updated, compact=True)
    assert store.load_replays(bundle) == updated
    assert path.read_bytes() == encode(updated, compact=True) != before
    assert b"\n " not in path.read_bytes() and path.read_bytes().endswith(b"\n")
    assert not path.with_suffix(".json.tmp").exists()
    assert pointer.read_bytes() == selected


def test_default_bytes_and_nonfinite_rejection():
    value = {"z": [1, 2], "a": "é"}
    assert (
        encode(value)
        == (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
    )
    assert (
        encode(value, compact=True)
        == (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    )
    for compact in (False, True):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError):
                encode({"bad": bad}, compact=compact)


def test_prediction_counts_keep_unsupported_losses_and_overlap(bundle):
    event = generate(1, 3)[0]
    supported = next(c for c in bundle.case_presets if c.segment == "growing")
    unsupported = Context(
        segment="restart", business_age_years=50.0, existing_debt_burden=1.4
    )
    rows = [
        event.model_copy(update={"context": c, "offer_id": "r900", "accepted": y})
        for c, y in [(supported, True), (unsupported, False)]
    ]
    gate = support_for(
        bundle.support, bundle.beta, unsupported, Finance(), "r900", "nov-2026", True
    )
    assert {"outside_training_range", "low_response_count"} <= set(gate.reasons)
    p = np.array([0.8, 0.9])
    score = prediction_score(
        rows, p, "fixture", "point", "fixture", "not_estimated", bundle=bundle
    )
    assert (
        score.advisory_supported_n,
        score.advisory_unsupported_n,
        score.missing_prediction_n,
        score.n,
        score.cohort_n,
    ) == (1, 1, 0, 2, 2)
    assert (score.brier, score.log_loss) == losses(np.array([1.0, 0.0]), p)
    assert sum(b.count for b in score.calibration) == 2
    raw = score.model_dump()
    for patch in (
        {"advisory_supported_n": -1},
        {"cohort_n": 3},
        {"advisory_unsupported_n": 0},
        {"calibration": []},
    ):
        with pytest.raises(ValidationError):
            PredictionScore.model_validate(raw | patch)
    for name in (
        "cohort_n",
        "missing_prediction_n",
        "support_definition",
        "advisory_supported_n",
        "advisory_unsupported_n",
    ):
        with pytest.raises(ValidationError):
            PredictionScore.model_validate({k: v for k, v in raw.items() if k != name})
    for p in (
        np.array([[0.5, 0.5]]),
        np.array([0.5]),
        np.array([np.nan, 0.5]),
        np.array([1.1, 0.5]),
    ):
        with pytest.raises(ValueError):
            prediction_score(
                rows, p, "fixture", "point", "fixture", "not_estimated", bundle=bundle
            )


@pytest.mark.parametrize("family", ["smooth", "nonlinear"])
def test_historical_confirmation_counts_from_full_family_evidence(bundle, family):
    # Historical worlds shared offered contexts/counts; only responses differed.
    # Preserve that historical fixture rather than force old counts onto the new complex DGP.
    fit = [e for e in generate(6000, 20261109, "smooth") if e.split == "fit"]
    test = [
        e
        for e in generate(6000, 20261110, "smooth", namespace="confirmation")
        if e.split == "test"
    ]
    family_bundle = bundle.model_copy(update={"beta": fit_beta(fit)})
    score = prediction_score(
        test,
        np.full(900, 0.5),
        family,
        "fixture",
        "historical repair",
        "not_estimated",
        bundle=family_bundle,
    )
    assert (
        score.advisory_supported_n,
        score.advisory_unsupported_n,
        score.missing_prediction_n,
        score.n,
    ) == (886, 14, 0, 900)
