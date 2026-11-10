"""Strict, safe metadata/results for one optional point-only offline challenger."""

from typing import Literal
from pydantic import model_validator

from lease_lab.contracts import Count, Finite, Hash, Provenance, StrictModel
from lease_lab.report_contracts import AuditFinding, PolicyScore, PredictionScore


class ChallengerInput(StrictModel):
    family: Literal["smooth", "nonlinear"]
    context_ids: list[str]
    freeze_sha256: Hash
    comparison_config_sha256: Hash
    feature_sha256: Hash
    context_matrix_sha256: Hash
    context_sha256: Hash
    confirmation_sha256: Hash
    fit_seconds: Finite
    prediction_seconds: Finite


class ChallengerReport(StrictModel):
    provenance: Provenance
    freeze_sha256: Hash
    comparison_config_sha256: Hash
    kind: Literal["tabular-transformer-point-benchmark"]
    library_version: str
    torch_version: str
    checkpoint_repo: str
    checkpoint_revision: str
    checkpoint_filename: str
    checkpoint_sha256: Hash
    license_url: str
    device: Literal["cpu"]
    context_applicants: Count
    estimators: Count
    uncertainty_method: Literal["not_estimated"]
    inputs: list[ChallengerInput]
    prediction: list[PredictionScore]
    policies: list[PolicyScore]
    findings: list[AuditFinding]
    elapsed_seconds: Finite
    peak_rss_bytes: Count

    @model_validator(mode="after")
    def coherent_contexts(self):
        if len(self.inputs) != 2 or {i.family for i in self.inputs} != {
            "smooth",
            "nonlinear",
        }:
            raise ValueError("both immutable simulator families required")
        if any(
            len(i.context_ids) != self.context_applicants
            or len(set(i.context_ids)) != self.context_applicants
            for i in self.inputs
        ):
            raise ValueError("whole distinct fit applicants required")
        if any(
            i.freeze_sha256 != self.freeze_sha256
            or i.comparison_config_sha256 != self.comparison_config_sha256
            for i in self.inputs
        ):
            raise ValueError("challenger input/experiment mismatch")
        for rows, suffix in ((self.prediction, "cohort"), (self.policies, "policy")):
            keys = [(r.family, r.model, getattr(r, suffix)) for r in rows]
            expected = {
                (
                    family,
                    "tabpfn-v2-512",
                    "sealed test" if suffix == "cohort" else "greedy",
                )
                for family in ("smooth", "nonlinear")
            }
            if len(keys) != 2 or set(keys) != expected:
                raise ValueError(
                    "exact TabPFN prediction and greedy inventory required"
                )
        return self
