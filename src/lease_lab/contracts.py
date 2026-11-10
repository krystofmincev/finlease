"""Strict public authority. Amounts are cents; rates nominal annual basis points."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Segment = Literal["new", "growing", "established", "restart"]
Prior = Literal["default", "weak", "strong"]
Method = Literal["beta", "laplace"]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        json_schema_serialization_defaults_required=True,
    )


class Context(StrictModel):
    segment: Segment = "established"
    business_age_years: Finite = Field(default=12.0, ge=0, le=60)
    monthly_turnover_cents: int = Field(default=3000000, ge=10000, le=200000000)
    existing_debt_burden: Finite = Field(default=0.18, ge=0, le=2)


class Finance(StrictModel):
    principal_cents: int = Field(default=3100000, ge=1, le=100000000)
    term_months: int = Field(default=60, ge=1, le=120)
    balloon_cents: int = Field(default=300000, ge=0)

    @model_validator(mode="after")
    def balloon_within_principal(self):
        if self.balloon_cents > self.principal_cents:
            raise ValueError("balloon exceeds financed principal")
        return self


class Provenance(StrictModel):
    schema_version: Literal["1.2"]
    origin: Literal["authored-fictional-simulation"]
    simulation_only: Literal[True]
    blueprint_version: Literal["1.1"]
    blueprint_sha256: Hash
    config_sha256: Hash
    data_sha256: Hash
    source_sha256: Hash
    run_id: str = Field(pattern=r"^[a-z0-9-]+$")
    seed: int
    currency: Literal["EUR"]
    money_unit: Literal["cent"]
    policy_version: Literal["fictional-policy-1.0"]


class DecisionRequest(StrictModel):
    context: Context = Field(default_factory=Context)
    finance: Finance = Field(default_factory=Finance)
    method: Method = "beta"
    prior: Prior = "default"
    card_id: Literal["nov-2026", "dec-2026"] = "nov-2026"
    as_of: str = Field(default="2026-11-09", pattern=r"^\d{4}-\d{2}-\d{2}$")
    draw_count: Literal[4096, 16384] = 4096
    draw_seed: int = Field(default=8128, ge=0, le=2147483647)

    @model_validator(mode="after")
    def valid_calendar_day(self):
        from datetime import date

        date.fromisoformat(self.as_of)
        return self


class Summary(StrictModel):
    mean: Finite
    q05: Finite
    q50: Finite
    q95: Finite

    @model_validator(mode="after")
    def ordered_quantiles(self):
        if not self.q05 <= self.q50 <= self.q95:
            raise ValueError("quantiles must be ordered")
        return self


class Support(StrictModel):
    state: Literal["supported", "sparse", "prior_only", "blocked"]
    automated: bool
    exposure_count: Count
    response_count: Count
    accepted_count: Count
    pending_count: Count
    observed_card_ids: list[str]
    observed_policy_versions: list[str]
    observed_rate_bps: list[int]
    within_training_range: bool
    reasons: list[str]

    @model_validator(mode="after")
    def coherent_support(self):
        if (
            self.exposure_count != self.response_count + self.pending_count
            or self.accepted_count > self.response_count
        ):
            raise ValueError("inconsistent evidence counts")
        if self.automated and (
            self.state != "supported"
            or self.reasons
            or not self.within_training_range
            or self.response_count == 0
        ):
            raise ValueError("automated advice requires consistent observed support")
        return self


class OfferResult(StrictModel):
    offer_id: str
    rate_bps: int
    eligible: bool
    monthly_payment_cents: int
    last_instalment_cents: int
    total_repayment_cents: int
    total_interest_cents: int
    contribution_if_accepted_cents: int
    decision_cost_cents: int
    support: Support
    acceptance: Summary
    expected_contribution_cents: Summary
    improvement_cents: Summary | None


class Choice(StrictModel):
    status: Literal["recommend", "retain", "review"]
    offer_id: str | None
    reason: str
    immediate_review_cost_cents: int = 0

    @model_validator(mode="after")
    def action_and_cost(self):
        if (self.status == "review") != (self.offer_id is None):
            raise ValueError("only review has no financing offer")
        if self.immediate_review_cost_cents < 0 or (
            self.status != "review" and self.immediate_review_cost_cents
        ):
            raise ValueError("review cost belongs only to a review")
        return self


class DecisionResult(StrictModel):
    provenance: Provenance
    request: DecisionRequest
    method: Method
    uncertainty_method: str
    uncertainty_target: str
    incumbent_offer_id: str | None
    greedy: Choice
    cautious: Choice
    offers: list[OfferResult]
    improvement_margin_cents: int
    numerical_tolerance_cents: int
    draw_count: int
    quantile_method: str
    diagnostics: list[str]
    elapsed_ms: Finite

    @model_validator(mode="after")
    def advice_is_supported(self):
        if len({o.offer_id for o in self.offers}) != len(self.offers):
            raise ValueError("duplicate offer result")
        for choice in (self.greedy, self.cautious):
            if choice.offer_id is not None and not any(
                o.offer_id == choice.offer_id and o.eligible and o.support.automated
                for o in self.offers
            ):
                raise ValueError("automated choice is not supported and eligible")
        for offer in self.offers:
            if not all(0 <= v <= 1 for v in offer.acceptance.model_dump().values()):
                raise ValueError("acceptance summary outside probability range")
        return self


class EvidenceCell(StrictModel):
    segment: Segment
    offer_id: str
    exposures: Count
    responses: Count
    accepted: Count
    alpha: Finite
    beta: Finite
    posterior: Summary
    support: Literal["supported", "sparse", "prior_only"]


class TeachingCell(StrictModel):
    accepted: Count
    observed: Count
    alpha: Finite
    beta: Finite
    posterior: Summary


class Metadata(StrictModel):
    provenance: Provenance
    available_methods: list[Method]
    evidence: list[EvidenceCell]
    teaching: list[TeachingCell]
    default_request: DecisionRequest
    minimum_responses: int
    stages: list[str]
    limitations: list[str]
    case_presets: list[Context]
