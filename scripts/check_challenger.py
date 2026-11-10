"""Read-only final comparison qualification; core-only mode is explicit repair evidence."""

import argparse
import json
import sys

from pydantic import ValidationError

from lease_lab.api.app import app
from lease_lab.artifacts.challenger import ChallengerReport
from lease_lab.artifacts.store import load_bundle, load_challenger, load_report
from lease_lab.config import CONFIG, ROOT
from lease_lab.contracts import DecisionRequest
from lease_lab.report_contracts import require_completed_comparison

parser = argparse.ArgumentParser()
parser.add_argument(
    "--core-only", action="store_true", help="unserved repair checkpoint only"
)
args = parser.parse_args()
bundle = load_bundle()
if args.core_only:
    core = load_report(bundle, include_challenger=False)
    assert core.comparison is None, (
        "core-only mode is reserved for the repair checkpoint"
    )
    print("Explicit repair-only core verified; no final comparison claim.")
    raise SystemExit(0)
report = load_challenger(bundle)  # Absence is a failed final gate.
merged = load_report(bundle)
require_completed_comparison(merged)
core_directory = ROOT / "artifacts/runs" / bundle.provenance.run_id
assert (
    sum(p.stat().st_size for p in core_directory.iterdir() if p.is_file())
    <= CONFIG["budget"]["artifact_megabytes"] * 1_000_000
)
assert (
    load_report(bundle, include_challenger=False).elapsed_seconds
    <= CONFIG["budget"]["offline_seconds"]
)
assert report.peak_rss_bytes <= 1_500_000_000
assert "torch" not in sys.modules and "tabpfn" not in sys.modules
assert report.uncertainty_method == "not_estimated" and report.device == "cpu"
split_ids = json.loads(
    (ROOT / "artifacts/runs" / bundle.provenance.run_id / "split-ids.json").read_bytes()
)["ids"]
for inputs in report.inputs:
    assert set(inputs.context_ids) <= set(split_ids["fit"])
    assert not set(inputs.context_ids) & set(split_ids["test"])
    assert len(inputs.context_ids) == report.context_applicants == 512
assert all(
    s.n == 900 and s.uncertainty_method == "not_estimated" for s in report.prediction
)
for patch in (
    {"uncertainty_method": "exact-posterior"},
    {"oracle_probabilities": [0.9]},
):
    try:
        ChallengerReport.model_validate(report.model_dump() | patch)
    except ValidationError:
        pass
    else:
        raise AssertionError("forbidden posterior/truth claim accepted")
for method in ("tabpfn-v2-512", "gbdt-point-selected", "logistic-point-512"):
    try:
        DecisionRequest(method=method)
    except ValidationError:
        pass
    else:
        raise AssertionError("point method entered posterior adapter")
assert app.openapi()["paths"].get("/api/challenger")
print(
    "Completed exact five-method prediction/greedy inventories, freeze/cohort/label budgets and serving isolation passed."
)
