"""Learner sees only chosen factual assignments and available feedback; no oracle."""

import numpy as np

from lease_lab.config import CONFIG, HISTORICAL_OFFERS
from lease_lab.contracts import Context
from lease_lab.data.events import Event
from lease_lab.models.beta_cells import BetaSampler, beta_summary, cell_for, fit_beta
from lease_lab.models.logistic_laplace import LogisticSampler, fit_logistic


class OnlineLearner:
    def __init__(self, initial: list[Event], method: str):
        if any(
            e.split != "replay" or e.accepted is None or e.available_at > 0
            for e in initial
        ):
            raise ValueError(
                "initial replay evidence must already be mature at clock zero"
            )
        self.history = list(initial)
        self.method = method
        self.beta = fit_beta(initial, clock=0)
        self.assignments = {e.applicant_id: e for e in initial}
        self.seen = {e.applicant_id for e in initial}
        self.state = fit_logistic(initial) if method == "laplace" else None
        self.sampler = (
            LogisticSampler(self.state) if self.state else BetaSampler(self.beta)
        )
        self.last_refit = len(initial)

    def assign(self, event: Event) -> None:
        if (
            event.accepted is not None
            or event.applicant_id in self.assignments
            or event.split != "replay"
        ):
            raise ValueError("one unseen chosen action with no future outcome required")
        self.assignments[event.applicant_id] = event
        self._count(event, pending=1, exposure=1)

    def observe(self, event: Event, clock: int) -> None:
        assigned = self.assignments.get(event.applicant_id)
        if (
            event.accepted is None
            or event.available_at > clock
            or event.applicant_id in self.seen
        ):
            raise ValueError("only new matured feedback can update the learner")
        if (
            assigned is None
            or event.offer_id != assigned.offer_id
            or event.context != assigned.context
            or event.decision_at != assigned.decision_at
            or event.outcome_at != assigned.outcome_at
            or event.available_at != assigned.available_at
        ):
            raise ValueError("feedback does not match a chosen factual assignment")
        self.seen.add(event.applicant_id)
        self.history.append(event)
        self._count(
            event,
            pending=-1,
            accepted=int(event.accepted),
            refused=int(not event.accepted),
        )

    def _count(
        self,
        event: Event,
        pending: int = 0,
        exposure: int = 0,
        accepted: int = 0,
        refused: int = 0,
    ):
        index = next(
            i
            for i, c in enumerate(self.beta.cells)
            if c.segment == event.context.segment and c.offer_id == event.offer_id
        )
        c = self.beta.cells[index]
        self.beta.cells[index] = type(c)(
            segment=c.segment,
            offer_id=c.offer_id,
            exposures=c.exposures + exposure,
            accepted=c.accepted + accepted,
            refused=c.refused + refused,
            pending=c.pending + pending,
        )

    def choose(self, context: Context, policy: str, seed: int) -> str:
        if (
            self.state
            and len(self.history) - self.last_refit >= CONFIG["replay_refit_interval"]
        ):
            self.state = fit_logistic(self.history, transform=self.state.transform)
            self.sampler = LogisticSampler(self.state)
            self.last_refit = len(self.history)
        if policy == "fixed":
            return "r900"
        if policy == "thompson":
            p = self.sampler.sample(context, HISTORICAL_OFFERS, 1, seed).probabilities[
                0
            ]
        elif policy == "greedy":
            p = (
                self.sampler.predict([context] * 3, HISTORICAL_OFFERS)[0]
                if self.state
                else np.array(
                    [
                        (cell_for(self.beta, context.segment, o).accepted + 1)
                        / (
                            cell_for(self.beta, context.segment, o).accepted
                            + cell_for(self.beta, context.segment, o).refused
                            + 2
                        )
                        for o in HISTORICAL_OFFERS
                    ]
                )
            )
        else:
            raise ValueError("unknown replay policy")
        # All replay policies have the same authorised exploration menu, including
        # prior-only cells. Advisory support gates are deliberately a separate mode.
        utility = (
            p
            * np.array(
                [CONFIG["acceptance_contribution_cents"][o] for o in HISTORICAL_OFFERS]
            )
            - CONFIG["decision_cost_cents"]
        )
        return HISTORICAL_OFFERS[int(np.argmax(utility))]

    def width(self) -> float:
        probe = Context()
        if self.state:
            _, low, high = self.sampler.predict([probe] * 3, HISTORICAL_OFFERS)
            return float(np.mean(high - low))
        summaries = [
            beta_summary(
                cell_for(self.beta, probe.segment, o).accepted + 1,
                cell_for(self.beta, probe.segment, o).refused + 1,
            )
            for o in HISTORICAL_OFFERS
        ]
        return float(np.mean([s.q95 - s.q05 for s in summaries]))

    def counts(self) -> tuple[list[int], list[int]]:
        return (
            [
                sum(c.exposures for c in self.beta.cells if c.offer_id == o)
                for o in HISTORICAL_OFFERS
            ],
            [
                sum(c.accepted + c.refused for c in self.beta.cells if c.offer_id == o)
                for o in HISTORICAL_OFFERS
            ],
        )
