import numpy as np
from scipy.stats import beta as beta_distribution

from lease_lab.config import CONFIG, HISTORICAL_OFFERS, PRIOR_BETA, SEGMENTS
from pydantic import model_validator

from lease_lab.contracts import (
    Context,
    Count,
    EvidenceCell,
    Segment,
    StrictModel,
    Summary,
)
from lease_lab.data.events import Event, mature_rows, require_binary_outcomes
from lease_lab.models.sampling import Draws


class Cell(StrictModel):
    segment: Segment
    offer_id: str
    exposures: Count
    accepted: Count
    refused: Count
    pending: Count

    @model_validator(mode="after")
    def coherent_counts(self):
        if self.accepted + self.refused + self.pending != self.exposures:
            raise ValueError("responses plus pending must equal exposures")
        if self.offer_id not in CONFIG["acceptance_contribution_cents"]:
            raise ValueError("unknown offer")
        return self


class BetaState(StrictModel):
    cells: list[Cell]

    @model_validator(mode="after")
    def complete_unique_cells(self):
        keys = [(c.segment, c.offer_id) for c in self.cells]
        expected = {(s, o) for s in SEGMENTS for o in HISTORICAL_OFFERS}
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("one cell per declared segment/offer required")
        return self


def fit_beta(events: list[Event], clock: int | None = None) -> BetaState:
    if any(
        e.split not in ("fit", "replay") or e.offer_id not in HISTORICAL_OFFERS
        for e in events
    ):
        raise ValueError(
            "Beta fitting requires fit/replay rows from the historical menu"
        )
    require_binary_outcomes(mature_rows(events, clock))
    cells = []
    for segment in SEGMENTS:
        for offer_id in HISTORICAL_OFFERS:
            rows = [
                e
                for e in events
                if e.context.segment == segment and e.offer_id == offer_id
            ]
            matured = mature_rows(rows, clock)
            cells.append(
                Cell(
                    segment=segment,
                    offer_id=offer_id,
                    exposures=len(rows),
                    accepted=sum(e.accepted is True for e in matured),
                    refused=sum(e.accepted is False for e in matured),
                    pending=len(rows) - len(matured),
                )
            )
    return BetaState(cells=cells)


def cell_for(state: BetaState, segment: str, offer_id: str) -> Cell:
    # New menu rates have no fitted cell. Return a zero-count diagnostic without
    # pretending they were part of the fitted historical state.
    matched = next(
        (c for c in state.cells if c.segment == segment and c.offer_id == offer_id),
        None,
    )
    if matched is not None:
        return matched
    return Cell(
        segment=segment,
        offer_id=offer_id,
        exposures=0,
        accepted=0,
        refused=0,
        pending=0,
    )


def beta_summary(alpha: float, beta: float) -> Summary:
    quantiles = beta_distribution.ppf([0.05, 0.5, 0.95], alpha, beta)
    return Summary(
        mean=alpha / (alpha + beta),
        q05=float(quantiles[0]),
        q50=float(quantiles[1]),
        q95=float(quantiles[2]),
    )


class BetaSampler:
    def __init__(self, state: BetaState, prior: str = "default"):
        self.state, self.prior = state, prior

    def sample(
        self, context: Context, offer_ids: tuple[str, ...], count: int, seed: int
    ) -> Draws:
        alpha, beta = PRIOR_BETA[self.prior]
        rng = np.random.default_rng(seed)
        columns = []
        for offer_id in offer_ids:
            cell = cell_for(self.state, context.segment, offer_id)
            columns.append(rng.beta(alpha + cell.accepted, beta + cell.refused, count))
        return Draws(
            np.column_stack(columns),
            offer_ids,
            "exact-independent-beta-cells",
            "cohort-average acceptance probability",
            (
                "Exact under independent-cell Bernoulli assumptions; not a personalised interval.",
            ),
        )


def evidence(
    state: BetaState, prior: str = "default", threshold: int = 30
) -> list[EvidenceCell]:
    a, b = PRIOR_BETA[prior]
    return [
        EvidenceCell(
            segment=c.segment,
            offer_id=c.offer_id,
            exposures=c.exposures,
            responses=c.accepted + c.refused,
            accepted=c.accepted,
            alpha=a + c.accepted,
            beta=b + c.refused,
            posterior=beta_summary(a + c.accepted, b + c.refused),
            support="prior_only"
            if c.accepted + c.refused == 0
            else "sparse"
            if c.accepted + c.refused < threshold
            else "supported",
        )
        for c in state.cells
    ]
