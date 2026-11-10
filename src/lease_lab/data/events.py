"""Factual, allowlisted learner input. Future/pending responses stay absent."""

from typing import Literal

from pydantic import Field, model_validator

from lease_lab.contracts import Context, Probability, StrictModel


class Event(StrictModel):
    applicant_id: str
    split: Literal["fit", "diagnostic", "selection", "test", "replay"]
    context: Context
    offer_id: str
    card_id: str = "nov-2026"
    policy_version: str = "fictional-policy-1.0"
    assignment_probability: Probability | None
    decision_at: int = Field(ge=0)
    outcome_at: int = Field(ge=0)
    available_at: int = Field(ge=0)
    accepted: bool | None

    @model_validator(mode="after")
    def ordering(self):
        if not self.decision_at <= self.outcome_at <= self.available_at:
            raise ValueError("decision <= occurrence <= availability required")
        if self.assignment_probability is not None and self.assignment_probability == 0:
            raise ValueError("chosen action cannot have zero assignment probability")
        return self


def mature_rows(events: list[Event], clock: int | None = None) -> list[Event]:
    return [
        e
        for e in events
        if e.accepted is not None and (clock is None or e.available_at <= clock)
    ]


def require_binary_outcomes(events: list[Event]) -> None:
    if {e.accepted for e in events if e.accepted is not None} != {False, True}:
        raise ValueError(
            "acceptance_not_estimable: both offered acceptances and refusals required; accepted-only data rejected"
        )
