"""Safe JSON codec, fixed project paths and content checks. No pickle/evaluator imports."""

import hashlib
import json
from pathlib import Path

from pydantic import Field, model_validator

from lease_lab.config import CONFIG_PATH, ROOT, SEGMENTS
from lease_lab.contracts import Context, Finite, Provenance, Segment, StrictModel
from lease_lab.models.beta_cells import BetaState
from lease_lab.models.logistic_laplace import LogisticState
from lease_lab.report_contracts import (
    EvaluationReport,
    ReplayReport,
    SelectionCheckpoint,
    require_completed_comparison,
    selection_payload,
)
from lease_lab.artifacts.challenger import ChallengerReport


class Range(StrictModel):
    minimum: Finite
    maximum: Finite

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("reversed training range")
        return self


class FeatureBounds(StrictModel):
    business_age_years: Range
    monthly_turnover_cents: Range
    existing_debt_burden: Range


class SupportProfile(FeatureBounds):
    observed_card_ids: list[str]
    observed_policy_versions: list[str]
    observed_rate_bps: list[int]
    by_segment: dict[Segment, FeatureBounds]

    @model_validator(mode="after")
    def complete_evidence_ranges(self):
        if set(self.by_segment) != set(SEGMENTS):
            raise ValueError("one fitted range profile per segment required")
        if (
            self.observed_card_ids != ["nov-2026"]
            or self.observed_policy_versions != ["fictional-policy-1.0"]
            or self.observed_rate_bps != [800, 900, 1000]
        ):
            raise ValueError("unknown historical support identity")
        return self


class FrozenBundle(StrictModel):
    provenance: Provenance
    beta: BetaState
    support: SupportProfile
    stages: list[str]
    laplace: dict[str, LogisticState] = Field(default_factory=dict)
    case_presets: list[Context]

    @model_validator(mode="after")
    def complete_presets(self):
        if len(self.case_presets) != len(SEGMENTS) or {
            c.segment for c in self.case_presets
        } != set(SEGMENTS):
            raise ValueError("one case preset per segment required")
        return self


class Current(StrictModel):
    run_id: str = Field(pattern=r"^[a-z0-9-]+$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: StrictModel | dict | list, *, compact: bool = False) -> bytes:
    if isinstance(value, StrictModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def write_json(
    path: Path, value: StrictModel | dict | list, *, compact: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encode(value, compact=compact))
    temporary.replace(path)


def source_digest() -> str:
    entries = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "configs").glob("*.json")
    ):
        entries.append(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest(b"\n".join(entries))


def publish_bundle(bundle: FrozenBundle, activate: bool = True) -> Path:
    directory = ROOT / "artifacts/runs" / bundle.provenance.run_id
    write_json(directory / "bundle.json", bundle)
    if activate:
        write_json(
            ROOT / "artifacts/current.json",
            Current(
                run_id=bundle.provenance.run_id,
                bundle_sha256=digest((directory / "bundle.json").read_bytes()),
            ),
        )
    return directory


def load_bundle() -> FrozenBundle:
    current = Current.model_validate_json(
        (ROOT / "artifacts/current.json").read_bytes()
    )
    path = ROOT / "artifacts/runs" / current.run_id / "bundle.json"
    raw = path.read_bytes()
    if digest(raw) != current.bundle_sha256:
        raise ValueError("frozen bundle digest mismatch")
    bundle = FrozenBundle.model_validate_json(raw)
    if (
        bundle.provenance.run_id != current.run_id
        or bundle.provenance.config_sha256 != digest(CONFIG_PATH.read_bytes())
    ):
        raise ValueError("inconsistent bundle/config identity; rebuild offline")
    return bundle


def load_report(
    bundle: FrozenBundle, include_challenger: bool = True
) -> EvaluationReport:
    path = ROOT / "artifacts/runs" / bundle.provenance.run_id / "evaluation.json"
    report = EvaluationReport.model_validate_json(path.read_bytes())
    if report.provenance != bundle.provenance:
        raise ValueError("report/bundle provenance mismatch")
    if report.comparison is not None:
        validate_selection(bundle, report)
    extra_path = (
        ROOT / "artifacts/challengers" / bundle.provenance.run_id / "tabpfn-v2.json"
    )
    if include_challenger and extra_path.exists():
        extra = load_challenger(bundle)
        methods = [
            m.model_copy(
                update={
                    "status": "completed",
                    "fit_seconds": next(
                        i.fit_seconds for i in extra.inputs if i.family == m.family
                    ),
                    "prediction_seconds": next(
                        i.prediction_seconds
                        for i in extra.inputs
                        if i.family == m.family
                    ),
                }
            )
            if m.model == "tabpfn-v2-512"
            else m
            for m in report.comparison.methods
        ]
        report = EvaluationReport.model_validate(
            report.model_dump()
            | {
                "comparison": report.comparison.model_copy(
                    update={"status": "completed", "methods": methods}
                ),
                "prediction": report.prediction + extra.prediction,
                "policies": report.policies + extra.policies,
                "findings": report.findings + extra.findings,
                "commands": report.commands
                + ["uv run --extra challenger lease-lab benchmark-tabpfn"],
                "interpretation": report.interpretation
                + [
                    "TabPFN V2 uses 512 context labels; epistemic uncertainty is not estimated. Its probability outputs never enter P5 or Thompson sampling. Built with PriorLabs-TabPFN."
                ],
                "elapsed_seconds": report.elapsed_seconds + extra.elapsed_seconds,
            }
        )
        require_completed_comparison(report)
    return report


def validate_selection(
    bundle: FrozenBundle, report: EvaluationReport
) -> SelectionCheckpoint:
    checkpoint = SelectionCheckpoint.model_validate_json(
        (
            ROOT / "artifacts/runs" / bundle.provenance.run_id / "selection.json"
        ).read_bytes()
    )
    comparison = report.comparison
    if (
        checkpoint.provenance != bundle.provenance
        or checkpoint.comparison.freeze_sha256 != comparison.freeze_sha256
    ):
        raise ValueError("selection/bundle identity mismatch")
    current = checkpoint.model_copy(update={"comparison": comparison})
    if digest(encode(selection_payload(current))) != comparison.freeze_sha256:
        raise ValueError("selection decisions changed after freeze")
    if comparison.config_sha256 != digest(
        (ROOT / "configs/point-comparison-v1.json").read_bytes()
    ):
        raise ValueError("comparison configuration mismatch")
    for family in ("smooth", "nonlinear"):
        roles = checkpoint.roles[family]
        dataset = next(d for d in report.datasets if d.family == family)
        if checkpoint.data_sha256[family] != dataset.data_sha256:
            raise ValueError("historical data identity mismatch")
        primary = [
            i
            for k in ("fit", "diagnostic", "selection", "historical_test")
            for i in roles[k]
        ]
        if (
            len(primary) != len(set(primary))
            or set(roles["early_stop"]) & set(roles["report_only"])
            or set(roles["early_stop"] + roles["report_only"])
            != set(roles["diagnostic"])
        ):
            raise ValueError("invalid development partition")
        for method in (m for m in comparison.methods if m.family == family):
            fit_ids = roles["fit_3600" if method.selection == "selected" else "fit_512"]
            if method.budget.fit.ids_sha256 != digest(
                encode(fit_ids)
            ) or method.budget.fit.n != len(fit_ids):
                raise ValueError("model fitting IDs differ from commitment")
    return checkpoint


def validate_challenger(extra: ChallengerReport, report: EvaluationReport) -> None:
    comparison = report.comparison
    if (
        comparison is None
        or extra.freeze_sha256 != comparison.freeze_sha256
        or extra.comparison_config_sha256 != comparison.config_sha256
    ):
        raise ValueError("challenger selection/experiment mismatch")
    for item in extra.inputs:
        method = next(
            m
            for m in comparison.methods
            if m.family == item.family and m.model == "tabpfn-v2-512"
        )
        dataset = next(d for d in report.datasets if d.family == item.family)
        if (
            item.confirmation_sha256 != dataset.confirmation_data_sha256
            or item.freeze_sha256 != comparison.freeze_sha256
            or item.comparison_config_sha256 != comparison.config_sha256
            or digest(encode(item.context_ids)) != method.budget.fit.ids_sha256
            or item.context_sha256 != method.fit_data_sha256
            or item.context_matrix_sha256 != method.fit_matrix_sha256
            or item.feature_sha256 != method.feature.transform_sha256
        ):
            raise ValueError("challenger cohort/context/features/freeze mismatch")
    if any(
        p.model != "tabpfn-v2-512" or p.cohort != "sealed test"
        for p in extra.prediction
    ) or any(
        p.model != "tabpfn-v2-512" or p.policy != "greedy" for p in extra.policies
    ):
        raise ValueError("challenger may contribute only committed TabPFN rows")


def load_challenger(bundle: FrozenBundle) -> ChallengerReport:
    path = ROOT / "artifacts/challengers" / bundle.provenance.run_id / "tabpfn-v2.json"
    report = ChallengerReport.model_validate_json(path.read_bytes())
    if report.provenance != bundle.provenance:
        raise ValueError("challenger/bundle provenance mismatch")
    base = EvaluationReport.model_validate_json(
        (
            ROOT / "artifacts/runs" / bundle.provenance.run_id / "evaluation.json"
        ).read_bytes()
    )
    validate_challenger(report, base)
    return report


def load_replays(bundle: FrozenBundle) -> ReplayReport:
    path = ROOT / "artifacts/runs" / bundle.provenance.run_id / "replays.json"
    report = ReplayReport.model_validate_json(path.read_bytes())
    if report.provenance != bundle.provenance:
        raise ValueError("replay/bundle provenance mismatch")
    return report
