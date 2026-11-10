"""Shared-draw expected values; no Bernoulli outcome noise or evaluator imports."""

import time

import numpy as np

from lease_lab.artifacts.store import FrozenBundle
from lease_lab.config import CONFIG
from lease_lab.contracts import (
    Choice,
    DecisionRequest,
    DecisionResult,
    OfferResult,
    Summary,
)
from lease_lab.decisions.support import support_for
from lease_lab.domain.finance import payment_schedule
from lease_lab.domain.menus import menu
from lease_lab.models.beta_cells import BetaSampler
from lease_lab.models.logistic_laplace import LogisticSampler, posterior_is_usable
from lease_lab.models.sampling import Sampler


def summarise(values: np.ndarray) -> Summary:
    quantiles = np.quantile(values, [0.05, 0.5, 0.95], method="linear")
    return Summary(
        mean=float(np.mean(values)),
        q05=float(quantiles[0]),
        q50=float(quantiles[1]),
        q95=float(quantiles[2]),
    )


def review(reason: str) -> Choice:
    return Choice(
        status="review",
        offer_id=None,
        reason=reason,
        immediate_review_cost_cents=CONFIG["review_cost_cents"],
    )


def tie_break(rows: list[OfferResult], score) -> OfferResult:
    maximum = max(score(row) for row in rows)
    ties = [
        row for row in rows if maximum - score(row) <= CONFIG["tie_tolerance_cents"]
    ]
    return min(ties, key=lambda row: (row.total_repayment_cents, row.offer_id))


def choose(
    offers: list[OfferResult], incumbent_id: str | None
) -> tuple[Choice, Choice]:
    supported = [o for o in offers if o.eligible and o.support.automated]
    if not supported:
        return review("insufficient_observed_support"), review(
            "insufficient_observed_support"
        )
    greedy_offer = tie_break(
        supported, lambda row: row.expected_contribution_cents.mean
    )
    greedy = Choice(
        status="retain" if greedy_offer.offer_id == incumbent_id else "recommend",
        offer_id=greedy_offer.offer_id,
        reason="highest_posterior_mean_contribution",
    )
    incumbent = next((o for o in supported if o.offer_id == incumbent_id), None)
    if incumbent is None:
        return greedy, review("incumbent_unavailable_or_unsupported")
    challengers = [
        o
        for o in supported
        if o.offer_id != incumbent_id and o.improvement_cents is not None
    ]
    retain = Choice(
        status="retain",
        offer_id=incumbent_id,
        reason="insufficient_lower_bound_improvement",
    )
    if not challengers:
        return greedy, retain
    best = tie_break(challengers, lambda row: row.improvement_cents.q05)
    lower = best.improvement_cents.q05
    margin, tolerance = (
        CONFIG["improvement_margin_cents"],
        CONFIG["quantile_boundary_tolerance_cents"],
    )
    if abs(lower - margin) <= tolerance:
        return greedy, review("numerically_borderline_improvement")
    if lower > margin:
        return greedy, Choice(
            status="recommend",
            offer_id=best.offer_id,
            reason="supported_lower_bound_exceeds_margin",
        )
    return greedy, retain


def evaluate(
    request: DecisionRequest, bundle: FrozenBundle, sampler: Sampler | None = None
) -> DecisionResult:
    started = time.perf_counter()
    offers, eligible = menu(request.card_id, request.as_of)
    ids = tuple(o.offer_id for o in offers)
    if sampler is None:
        if request.method == "laplace":
            if request.prior not in bundle.laplace:
                raise ValueError(
                    "requested model is not available in this frozen bundle"
                )
            state = bundle.laplace[request.prior]
            sampler = (
                LogisticSampler(state)
                if posterior_is_usable(state.diagnostics)
                else BetaSampler(bundle.beta, request.prior)
            )
        else:
            sampler = BetaSampler(bundle.beta, request.prior)
    draws = sampler.sample(request.context, ids, request.draw_count, request.draw_seed)
    if draws.offer_ids != ids or draws.probabilities.shape != (
        request.draw_count,
        len(ids),
    ):
        raise ValueError("adapter violated joint draw/order contract")
    probabilities = draws.probabilities
    utility = probabilities * np.array(
        [o.contribution_cents for o in offers]
    ) - np.array([o.cost_cents for o in offers])
    incumbent_id = (
        CONFIG["incumbent_offer_id"]
        if CONFIG["incumbent_offer_id"] in ids and eligible
        else None
    )
    incumbent_index = ids.index(incumbent_id) if incumbent_id else None
    results = []
    for i, offer in enumerate(offers):
        schedule = payment_schedule(request.finance, offer.rate_bps)
        results.append(
            OfferResult(
                offer_id=offer.offer_id,
                rate_bps=offer.rate_bps,
                eligible=eligible,
                monthly_payment_cents=schedule.monthly_cents,
                last_instalment_cents=schedule.last_instalment_cents,
                total_repayment_cents=schedule.total_cents,
                total_interest_cents=schedule.interest_cents,
                contribution_if_accepted_cents=offer.contribution_cents,
                decision_cost_cents=offer.cost_cents,
                support=support_for(
                    bundle.support,
                    bundle.beta,
                    request.context,
                    request.finance,
                    offer.offer_id,
                    request.card_id,
                    eligible,
                ),
                acceptance=summarise(probabilities[:, i]),
                expected_contribution_cents=summarise(utility[:, i]),
                improvement_cents=summarise(utility[:, i] - utility[:, incumbent_index])
                if incumbent_index is not None
                else None,
            )
        )
    greedy, cautious = choose(results, incumbent_id)
    actual_method = (
        "laplace" if draws.method == "gaussian-prior-map-laplace" else "beta"
    )
    diagnostics = list(draws.diagnostics)
    if request.method != actual_method:
        diagnostics.append(
            "Requested approximation failed its numerical fit gate; exact-cell fallback used."
        )
    return DecisionResult(
        provenance=bundle.provenance,
        request=request,
        method=actual_method,
        uncertainty_method=draws.method,
        uncertainty_target=draws.target,
        incumbent_offer_id=incumbent_id,
        greedy=greedy,
        cautious=cautious,
        offers=results,
        improvement_margin_cents=CONFIG["improvement_margin_cents"],
        numerical_tolerance_cents=CONFIG["quantile_boundary_tolerance_cents"],
        draw_count=request.draw_count,
        quantile_method="numpy linear empirical quantiles; shared parameter draws",
        diagnostics=diagnostics,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
