"""Read frozen results to exercise the real cross-language response contract."""

from lease_lab.api.app import metadata
from lease_lab.artifacts.store import (
    load_bundle,
    load_report,
    load_replays,
    write_json,
    load_challenger,
)
from lease_lab.config import ROOT
from lease_lab.contracts import DecisionRequest
from lease_lab.decisions.policy import evaluate

b = load_bundle()
fixtures = {
    "Metadata": metadata().model_dump(mode="json"),
    "DecisionResult": evaluate(DecisionRequest(), b).model_dump(mode="json"),
    "EvaluationReport": load_report(b).model_dump(mode="json"),
    "ReplayReport": load_replays(b).model_dump(mode="json"),
}
try:
    fixtures["ChallengerReport"] = load_challenger(b).model_dump(mode="json")
except FileNotFoundError:
    pass
write_json(ROOT / ".local/contract-fixtures.json", fixtures)
print("Saved current real response fixtures; no fitting or simulation.")
