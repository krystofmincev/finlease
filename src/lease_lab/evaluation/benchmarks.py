"""Offline scoring of predeclared frozen methods. No tuning from evaluator truth."""

from dataclasses import replace
import numpy as np
from scipy.special import expit
from scipy.stats import norm

from lease_lab.artifacts.store import FrozenBundle
from lease_lab.config import CONFIG, HISTORICAL_OFFERS
from lease_lab.contracts import DecisionRequest, Finance
from lease_lab.data.events import Event
from lease_lab.decisions.policy import evaluate, review
from lease_lab.decisions.support import support_for
from lease_lab.evaluation.metrics import paired_bootstrap, prediction_score
from lease_lab.evaluation.references import (
    compare_quantiles,
    integrate_reference,
    reference_fixtures,
)
from lease_lab.models.beta_cells import cell_for
from lease_lab.models.logistic_laplace import (
    LogisticSampler,
    fit_map,
    sample_coefficients,
)
from lease_lab.report_contracts import NumericalCheck, PolicyScore, PredictionScore
from lease_lab.simulation.environment import probability


def numerical_checks() -> list[NumericalCheck]:
    checks = []
    for fixture in reference_fixtures():
        reference = integrate_reference(
            fixture.X, fixture.y, fixture.prior_sd, fixture.probes
        )
        scale = fixture.prior_sd / 3.0
        fitted = fit_map(fixture.X * scale, fixture.y, 3.0)
        probes = fixture.probes * scale
        mean = probes @ fitted.mode
        variance = np.einsum("ij,jk,ik->i", probes, fitted.covariance, probes)
        quantiles = expit(
            mean[:, None] + np.sqrt(variance)[:, None] * norm.ppf([0.05, 0.5, 0.95])
        )
        comparison = compare_quantiles(reference, quantiles)
        if fixture.category == "ordinary" and not comparison.passed:
            raise ValueError("ordinary Laplace reference gate failed: " + fixture.name)
        fallback = None
        if not comparison.passed:
            rejected = replace(
                fitted,
                diagnostics=fitted.diagnostics.model_copy(
                    update={"reference_check": "failed"}
                ),
            )
            try:
                sample_coefficients(rejected, 1, 7)
            except ValueError:
                fallback = review("approximation_reference_failed")
            else:
                raise AssertionError("failed approximation was sampled")
        checks.append(
            NumericalCheck(
                fixture=fixture.name,
                category=fixture.category,
                dimensions=fixture.X.shape[1],
                max_quantile_error=comparison.max_absolute_error,
                tolerance=comparison.tolerance,
                posterior_tail_bound=reference.diagnostics.posterior_tail_probability_upper_bound,
                grid_quantile_difference=reference.diagnostics.max_quantile_grid_difference,
                status=comparison.status,
                disposition="Within declared tolerance for this fixture only."
                if comparison.passed
                else "Approximation rejected and review returned for this fixture. Not a universal per-case detector.",
                fallback=fallback,
            )
        )
    return checks


def predictions(
    rows: list[Event],
    fit: list[Event],
    bundle: FrozenBundle,
    family: str,
    cohort: str,
    *,
    point_predictions: np.ndarray,
) -> list[PredictionScore]:
    state = bundle.laplace["default"]
    rate_mean = {
        o: np.mean([e.accepted for e in fit if e.offer_id == o])
        for o in HISTORICAL_OFFERS
    }
    means = []
    empirical = []
    for e in rows:
        c = cell_for(bundle.beta, e.context.segment, e.offer_id)
        means.append((c.accepted + 1) / (c.accepted + c.refused + 2))
        empirical.append(
            c.accepted / (c.accepted + c.refused) if c.accepted + c.refused else 0.5
        )
    results = []
    estimates = [
        (
            "empirical-rate",
            np.array([rate_mean[e.offer_id] for e in rows]),
            "not_estimated",
        ),
        ("empirical-cell", np.asarray(empirical), "not_estimated"),
        ("beta", np.asarray(means), "exact-independent-beta-cells"),
        ("logistic-point", point_predictions, "not_estimated"),
        (
            "laplace",
            LogisticSampler(state).predict(
                [e.context for e in rows], [e.offer_id for e in rows]
            )[0],
            "map-laplace-approximation",
        ),
    ]
    for name, p, uncertainty in estimates:
        results.append(
            prediction_score(rows, p, family, name, cohort, uncertainty, bundle=bundle)
        )
    return results


def policy_scores(
    rows: list[Event], bundle: FrozenBundle, family: str
) -> list[PolicyScore]:
    result = []
    for method in ("beta", "laplace"):
        values = {
            name: []
            for name in (
                "incumbent",
                "fixed-8",
                "fixed-9",
                "fixed-10",
                "greedy",
                "cautious",
            )
        }
        acceptances = {name: [] for name in values}
        reviews = {name: 0 for name in values}
        violations = {name: 0 for name in values}
        for event in rows:
            decision = evaluate(
                DecisionRequest(context=event.context, method=method), bundle
            )
            # Fixed comparators explicitly include the advisory support fallback.
            fixed = {
                "incumbent": "r900",
                "fixed-8": "r800",
                "fixed-9": "r900",
                "fixed-10": "r1000",
            }
            choices = {
                name: chosen
                if any(
                    o.offer_id == chosen and o.eligible and o.support.automated
                    for o in decision.offers
                )
                else None
                for name, chosen in fixed.items()
            }
            choices.update(
                {
                    "greedy": decision.greedy.offer_id,
                    "cautious": decision.cautious.offer_id,
                }
            )
            for name, chosen in choices.items():
                row = next(
                    (o for o in decision.offers if o.offer_id == chosen),
                    None,
                )
                invalid_attempt = chosen is not None and (
                    row is None or not row.eligible or not row.support.automated
                )
                violations[name] += int(invalid_attempt)
                if row is None or invalid_attempt:
                    values[name].append(-CONFIG["review_cost_cents"])
                    acceptances[name].append(0.0)
                    reviews[name] += 1
                else:
                    p = probability(event.context, row.rate_bps, family)
                    values[name].append(
                        p * row.contribution_if_accepted_cents - row.decision_cost_cents
                    )
                    acceptances[name].append(p)
        incumbent = np.asarray(values["incumbent"])
        for name, numbers in values.items():
            delta = np.asarray(numbers) - incumbent
            low, high = paired_bootstrap(delta)
            result.append(
                PolicyScore(
                    family=family,
                    model=method,
                    policy=name,
                    n=len(rows),
                    reviews=reviews[name],
                    recommendation_coverage=1 - reviews[name] / len(rows),
                    expected_acceptance=float(np.mean(acceptances[name])),
                    mean_expected_contribution_cents=float(np.mean(numbers)),
                    review_handling_cost_cents=reviews[name]
                    * CONFIG["review_cost_cents"],
                    support_violations=violations[name],
                    paired_difference_cents=float(delta.mean()),
                    paired_bootstrap_low_cents=low,
                    paired_bootstrap_high_cents=high,
                )
            )
    return result


def point_policy(rows, probabilities, bundle, family, model, *, offer_ids):
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        tuple(offer_ids) != HISTORICAL_OFFERS
        or probabilities.shape != (len(rows), 3)
        or not len(rows)
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("finite [applicant, canonical offer] probabilities required")
    state = bundle.beta
    margins = np.array(
        [CONFIG["acceptance_contribution_cents"][o] for o in HISTORICAL_OFFERS]
    )
    actions = []
    supports = []
    for event, p in zip(rows, probabilities):
        allowed = [
            support_for(
                bundle.support,
                state,
                event.context,
                Finance(),
                o,
                event.card_id,
                event.card_id == "nov-2026",
            ).automated
            for o in HISTORICAL_OFFERS
        ]
        supports.append(allowed)
        actions.append(
            max(
                (j for j in range(3) if allowed[j]),
                key=lambda j: (p[j] * margins[j], -j),
            )
            if any(allowed)
            else None
        )
    # Independently score the returned action; do not erase invalid attempts before counting.
    values = []
    acceptances = []
    reviews = 0
    violations = 0
    incumbent = []
    for event, chosen, allowed in zip(rows, actions, supports):
        invalid = chosen is not None and not allowed[chosen]
        violations += int(invalid)
        if chosen is None or invalid:
            values.append(-CONFIG["review_cost_cents"])
            acceptances.append(0.0)
            reviews += 1
        else:
            p = probability(event.context, int(HISTORICAL_OFFERS[chosen][1:]), family)
            values.append(p * margins[chosen] - CONFIG["decision_cost_cents"])
            acceptances.append(p)
        incumbent.append(
            probability(event.context, 900, family) * margins[1]
            - CONFIG["decision_cost_cents"]
            if allowed[1]
            else -CONFIG["review_cost_cents"]
        )
    difference = np.asarray(values) - np.asarray(incumbent)
    low, high = paired_bootstrap(difference)
    return PolicyScore(
        family=family,
        model=model,
        policy="greedy",
        n=len(rows),
        reviews=reviews,
        recommendation_coverage=1 - reviews / len(rows),
        expected_acceptance=float(np.mean(acceptances)),
        mean_expected_contribution_cents=float(np.mean(values)),
        review_handling_cost_cents=reviews * CONFIG["review_cost_cents"],
        support_violations=violations,
        paired_difference_cents=float(difference.mean()),
        paired_bootstrap_low_cents=low,
        paired_bootstrap_high_cents=high,
    )
