"""Explicit public allowlist for saved aggregate evaluation and replay evidence."""

from typing import Literal
from pydantic import model_validator

from lease_lab.contracts import (
    Count,
    Finite,
    Hash,
    Probability,
    Provenance,
    StrictModel,
)
from lease_lab.contracts import Choice


class CalibrationBin(StrictModel):
    lower: Probability
    upper: Probability
    count: Count
    prediction: Probability | None
    observed: Probability | None


class SliceScore(StrictModel):
    slice: str
    n: Count
    brier: Finite
    log_loss: Finite


class PredictionScore(StrictModel):
    family: str
    model: str
    cohort: str
    n: Count
    cohort_n: Count
    advisory_supported_n: Count
    advisory_unsupported_n: Count
    missing_prediction_n: Count
    support_definition: Literal["advisory-support-heuristic-v1"]
    brier: Finite
    log_loss: Finite
    calibration_error: Finite
    calibration: list[CalibrationBin]
    slices: list[SliceScore]
    uncertainty_method: str

    @model_validator(mode="after")
    def counted_predictions(self):
        if self.advisory_supported_n + self.advisory_unsupported_n != self.n:
            raise ValueError("support counts must partition scored predictions")
        if self.n + self.missing_prediction_n != self.cohort_n:
            raise ValueError("scored and missing counts must partition cohort")
        if sum(b.count for b in self.calibration) != self.n:
            raise ValueError("calibration counts must sum to scored predictions")
        return self


class NumericalCheck(StrictModel):
    fixture: str
    category: str
    dimensions: int
    max_quantile_error: Finite
    tolerance: Finite
    posterior_tail_bound: Finite
    grid_quantile_difference: Finite
    status: Literal["pass", "reference_unreliable", "approximation_error"]
    disposition: str
    fallback: Choice | None


class CoverageRow(StrictModel):
    family: str
    model: str
    probe: str
    target: str
    support: str
    seeds: int
    covered: int
    coverage: Probability
    interval_low: Probability
    interval_high: Probability
    mean_width: Finite
    target_standard_error: Finite


class PolicyScore(StrictModel):
    family: str
    model: str
    policy: str
    n: Count
    reviews: Count
    recommendation_coverage: Probability
    expected_acceptance: Probability
    mean_expected_contribution_cents: Finite
    review_handling_cost_cents: int
    support_violations: Count
    paired_difference_cents: Finite
    paired_bootstrap_low_cents: Finite
    paired_bootstrap_high_cents: Finite


class AuditFinding(StrictModel):
    name: str
    status: str
    detail: str


class DatasetIdentity(StrictModel):
    family: str
    data_sha256: Hash
    split_ids_sha256: Hash
    applicants: int
    fit: int
    diagnostic: int
    selection: int
    test: int
    confirmation_seed: int
    confirmation_namespace: str
    confirmation_data_sha256: Hash


class FeatureIdentity(StrictModel):
    design: Literal["base", "rate-interactions", "without-burden"]
    columns: list[str]
    transform_sha256: Hash
    base_mean: list[Finite]
    base_scale: list[Finite]
    derived_mean: list[Finite]
    derived_scale: list[Finite]
    fit_context_n: Count
    precision: Literal["float32", "float64"]


class LabelRole(StrictModel):
    n: Count
    ids_sha256: Hash


class LabelBudget(StrictModel):
    fit: LabelRole
    early_stop: LabelRole
    selection: LabelRole
    calibration: LabelRole
    monitoring: LabelRole
    consumed_union_n: Count
    observed_union_n: Count
    overlap: list[str]
    preprocessing_context_n: Count
    support_history_n: Count
    external_pretraining: bool


class SelectionTrial(StrictModel):
    candidate_id: str
    design: Literal["base", "rate-interactions", "without-burden"]
    columns: Count
    feature_sha256: Hash
    parameters: dict[str, int | Finite | str | bool | None]
    status: Literal["completed", "failed", "budget_cutoff"]
    failure: str | None
    selection_log_loss: Finite | None
    selection_brier: Finite | None
    selection_calibration_error: Finite | None
    iterations: Count
    chosen_round: Count | None
    stopped_round: Count | None
    validation_stop: bool
    round_cap: bool
    reason: str
    fit_seconds: Finite
    train_loss: list[Finite]
    development_loss: list[Finite]
    elapsed_seconds: Finite

    @model_validator(mode="after")
    def complete_trial(self):
        metrics = (
            self.selection_log_loss,
            self.selection_brier,
            self.selection_calibration_error,
        )
        if self.status == "completed":
            if (
                any(v is None for v in metrics)
                or self.failure is not None
                or self.iterations == 0
            ):
                raise ValueError(
                    "completed candidate requires actual convergence and scores"
                )
            if self.chosen_round is not None and (
                self.stopped_round is None
                or not 1 <= self.chosen_round <= self.stopped_round
                or len(self.train_loss) != self.stopped_round + 1
                or len(self.development_loss) != self.stopped_round + 1
            ):
                raise ValueError(
                    "tree curve must include round zero and chosen positive prefix"
                )
        elif not self.failure or any(v is not None for v in metrics):
            raise ValueError(
                "failed candidate requires explicit cause and no scored criterion"
            )
        return self


class LearningCurvePoint(StrictModel):
    fit_n: Count
    fit_ids_sha256: Hash
    train_log_loss: Finite
    early_stop_log_loss: Finite
    iterations: Count
    elapsed_seconds: Finite


class MethodDevelopment(StrictModel):
    family: Literal["smooth", "nonlinear"]
    model: str
    status: Literal["completed", "precommitted"]
    selection: Literal["fixed", "selected", "pretrained"]
    feature: FeatureIdentity
    fit_matrix_sha256: Hash
    fit_data_sha256: Hash
    budget: LabelBudget
    parameters: dict[str, int | Finite | str | bool | None]
    optimizer: str
    chosen_candidate: str | None
    trials: list[SelectionTrial]
    curves: list[LearningCurvePoint]
    fit_seconds: Finite
    prediction_seconds: Finite


class PointComparison(StrictModel):
    version: Literal["point-comparison-v1"]
    scenario_versions: dict[str, str]
    config_sha256: Hash
    authority_version: Literal["2.2"]
    authority_sha256: Hash
    final_seed: int
    final_namespace: str
    freeze_sha256: Hash
    status: Literal["awaiting_tabpfn", "completed"]
    development_fit_count: Count
    development_seconds: Finite
    baseline_fit_count: Count
    methods: list[MethodDevelopment]


class SelectionCheckpoint(StrictModel):
    provenance: Provenance
    comparison: PointComparison
    roles: dict[str, dict[str, list[str]]]
    data_sha256: dict[str, Hash]
    frozen_before_final: Literal[True]
    pre_generation_run_id: str
    pre_generation_sha256: Hash | None


def selection_payload(checkpoint: SelectionCheckpoint) -> dict:
    """Stable decisions only; exclude stage IDs, status and measured timings."""
    comparison = checkpoint.comparison
    return {
        "source": checkpoint.provenance.source_sha256,
        "lab_config": checkpoint.provenance.config_sha256,
        "comparison_config": comparison.config_sha256,
        "authority": comparison.authority_sha256,
        "scenarios": comparison.scenario_versions,
        "final_seed": comparison.final_seed,
        "final_namespace": comparison.final_namespace,
        "roles": checkpoint.roles,
        "data": checkpoint.data_sha256,
        "methods": [
            m.model_dump(
                mode="json",
                exclude={
                    "status",
                    "fit_seconds",
                    "prediction_seconds",
                    "trials",
                    "curves",
                },
            )
            for m in comparison.methods
        ],
    }


class EvaluationReport(StrictModel):
    provenance: Provenance
    prediction: list[PredictionScore]
    numerical: list[NumericalCheck]
    coverage: list[CoverageRow]
    policies: list[PolicyScore]
    findings: list[AuditFinding]
    commands: list[str]
    elapsed_seconds: Finite
    interpretation: list[str]
    datasets: list[DatasetIdentity]
    comparison: PointComparison | None

    @model_validator(mode="after")
    def unique_results(self):
        for keys in (
            [(p.family, p.model, p.cohort) for p in self.prediction],
            [(p.family, p.model, p.policy) for p in self.policies],
        ):
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate model/cohort or model/policy result")
        return self


POINT_METHODS = (
    "logistic-point-512",
    "gbdt-point-512",
    "tabpfn-v2-512",
    "logistic-point-selected",
    "gbdt-point-selected",
)


def require_completed_comparison(report: EvaluationReport) -> None:
    comparison = report.comparison
    if comparison is None or comparison.status != "completed":
        raise ValueError("completed three-family comparison required")
    if (
        not 0 < comparison.development_fit_count <= 42
        or comparison.development_seconds > 60
    ):
        raise ValueError("development budget incomplete or exceeded")
    if len(comparison.methods) != 10 or {d.family for d in report.datasets} != {
        "smooth",
        "nonlinear",
    }:
        raise ValueError("both complete family inventories required")
    if any(p.model in POINT_METHODS and p.policy != "greedy" for p in report.policies):
        raise ValueError("point models cannot supply cautious or Thompson policies")
    for family in ("smooth", "nonlinear"):
        records = [m for m in comparison.methods if m.family == family]
        if len(records) != 5 or {m.model for m in records} != set(POINT_METHODS):
            raise ValueError("exact five-method development inventory required")
        controls = [m for m in records if m.model.endswith("512")]
        if (
            len(
                {
                    (
                        m.feature.transform_sha256,
                        m.fit_matrix_sha256,
                        m.fit_data_sha256,
                        m.budget.fit.ids_sha256,
                    )
                    for m in controls
                }
            )
            != 1
        ):
            raise ValueError("512 controls must share fitting IDs/data/base matrices")
        dataset = next(d for d in report.datasets if d.family == family)
        if (
            dataset.confirmation_seed != comparison.final_seed
            or dataset.confirmation_namespace != comparison.final_namespace
        ):
            raise ValueError("comparison/dataset recipe mismatch")
        for method in records:
            scores = [
                s
                for s in report.prediction
                if s.family == family
                and s.model == method.model
                and s.cohort == "sealed test"
            ]
            policies = [
                s
                for s in report.policies
                if s.family == family
                and s.model == method.model
                and s.policy == "greedy"
            ]
            if (
                method.status != "completed"
                or len(scores) != 1
                or len(policies) != 1
                or scores[0].n != dataset.test
                or policies[0].n != dataset.test
            ):
                raise ValueError(
                    "missing, duplicate or incomplete final prediction/greedy policy"
                )
            if scores[0].uncertainty_method != "not_estimated":
                raise ValueError("point method cannot claim model uncertainty")
            budget = method.budget
            selected = method.selection == "selected"
            if budget.monitoring.n != (
                450 if selected else 0
            ) or budget.external_pretraining != (method.model == "tabpfn-v2-512"):
                raise ValueError(
                    "monitoring or external-pretraining disclosure mismatch"
                )
            if budget.overlap != (
                ["early_stop=monitoring"]
                if method.model == "gbdt-point-selected"
                else []
            ):
                raise ValueError("stopping/monitoring overlap disclosure mismatch")
            expected = (
                3600 if selected else 512,
                450 if method.model == "gbdt-point-selected" else 0,
                600 if selected else 0,
            )
            if (
                (budget.fit.n, budget.early_stop.n, budget.selection.n) != expected
                or budget.calibration.n != 0
                or budget.preprocessing_context_n != 3600
                or budget.support_history_n != 3600
            ):
                raise ValueError("misstated comparison label roles")
            if (
                budget.consumed_union_n != sum(expected)
                or budget.observed_union_n
                != expected[0] + expected[2] + budget.monitoring.n
            ):
                raise ValueError("label unions disagree")
            if selected:
                expected_trials = 9 if method.model.startswith("logistic") else 5
                if (
                    len(method.trials) != expected_trials
                    or any(t.status == "budget_cutoff" for t in method.trials)
                    or not any(
                        t.candidate_id == method.chosen_candidate
                        and t.status == "completed"
                        for t in method.trials
                    )
                ):
                    raise ValueError("incomplete candidate accounting")
                if [c.fit_n for c in method.curves] != [512, 1800, 3600]:
                    raise ValueError("missing actual development curve")
                if len({t.candidate_id for t in method.trials}) != expected_trials:
                    raise ValueError("duplicate search candidate")
                if method.model.startswith("logistic"):
                    actual_grid = {(t.design, t.parameters["C"]) for t in method.trials}
                    expected_grid = {
                        (d, c)
                        for d in ("base", "rate-interactions", "without-burden")
                        for c in (0.1, 1.0, 10.0)
                    }
                else:
                    actual_grid = {
                        (
                            t.design,
                            t.parameters["learning_rate"],
                            t.parameters["max_leaf_nodes"],
                        )
                        for t in method.trials
                    }
                    expected_grid = {
                        ("base", r, leaves) for r in (0.03, 0.1) for leaves in (7, 15)
                    } | {("without-burden", 0.1, 7)}
                if actual_grid != expected_grid:
                    raise ValueError("incomplete declared candidate grid")


class Feedback(StrictModel):
    decision_index: Count
    offer_id: str
    outcome_at: Count
    available_at: Count
    accepted: bool


class ReplayFrame(StrictModel):
    index: Count
    chosen_offer_id: str
    decision_at: Count
    outcome_at: Count
    available_at: Count
    assignment_probability: None = None
    matured: Count
    pending: Count
    exposures: list[Count]
    responses: list[Count]
    mean_interval_width: Finite
    interval_as_of_index: Count
    cumulative_realised_cents: Finite
    cumulative_expected_cents: Finite
    cumulative_oracle_regret_cents: Finite
    feedback: list[Feedback]

    @model_validator(mode="after")
    def feedback_is_available(self):
        if (
            len(self.exposures) != 3
            or len(self.responses) != 3
            or any(r > e for r, e in zip(self.responses, self.exposures))
        ):
            raise ValueError("three coherent action exposure/response counts required")
        if (
            sum(self.responses) != self.matured
            or sum(self.exposures) != self.matured + self.pending
        ):
            raise ValueError("maturity and exposure totals disagree")
        if any(
            f.outcome_at > f.available_at or f.available_at > self.decision_at
            for f in self.feedback
        ):
            raise ValueError("future feedback cannot appear in replay frames")
        return self


class ReplayPath(StrictModel):
    model: str
    policy: str
    scenario: str
    seed: int
    frames: list[ReplayFrame]


class ReplaySummary(StrictModel):
    model: str
    policy: str
    scenario: str
    seeds: int
    mean_realised_cents: Finite
    mean_expected_cents: Finite
    expected_q05_cents: Finite
    expected_q95_cents: Finite
    mean_regret_cents: Finite
    paired_vs_fixed_mean_cents: Finite
    paired_vs_fixed_low_cents: Finite
    paired_vs_fixed_high_cents: Finite


class ReplayReport(StrictModel):
    provenance: Provenance
    paths: list[ReplayPath]
    summaries: list[ReplaySummary]
    scenarios: list[str]
    seed_ids: list[int]
    initial_evidence_count: int
    decisions_per_seed: int
    elapsed_seconds: Finite
    interpretation: list[str]
