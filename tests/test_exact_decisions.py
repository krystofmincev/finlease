import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.integrate import quad
from scipy.optimize import brentq

from lease_lab.api.app import app
from lease_lab.artifacts.store import load_bundle
from lease_lab.contracts import Context, DecisionRequest, Finance
from lease_lab.decisions.policy import choose, evaluate
from lease_lab.models.beta_cells import BetaSampler, beta_summary, cell_for


def test_beta_moments_and_independent_polynomial_quantile():
    for accepted, total in [(8, 10), (800, 1000)]:
        a, b = accepted + 1, total - accepted + 1
        summary = beta_summary(a, b)
        assert summary.mean == a / (a + b)
        if total == 10:
            normalizer = math.factorial(a + b - 1) / (
                math.factorial(a - 1) * math.factorial(b - 1)
            )
            cdf = lambda x: quad(
                lambda t: normalizer * t ** (a - 1) * (1 - t) ** (b - 1),
                0,
                x,
                epsabs=1e-12,
            )[0]
            for q, actual in [
                (0.05, summary.q05),
                (0.5, summary.q50),
                (0.95, summary.q95),
            ]:
                assert actual == pytest.approx(
                    brentq(lambda x: cdf(x) - q, 0, 1), abs=1e-10
                )
        samples = np.random.default_rng(52).beta(a, b, 200000)
        expected_variance = a * b / ((a + b) ** 2 * (a + b + 1))
        assert samples.var() == pytest.approx(expected_variance, rel=0.015)


def test_paired_differences_and_exact_cells():
    bundle = load_bundle()
    request = DecisionRequest()
    result = evaluate(request, bundle)
    ids = tuple(o.offer_id for o in result.offers)
    p = (
        BetaSampler(bundle.beta)
        .sample(request.context, ids, request.draw_count, request.draw_seed)
        .probabilities
    )
    u = p * np.array([o.contribution_if_accepted_cents for o in result.offers]) - 3500
    for i, offer in enumerate(result.offers):
        assert offer.improvement_cents.q05 == pytest.approx(
            np.quantile(u[:, i] - u[:, 1], 0.05)
        )
    assert result.offers[1].improvement_cents.model_dump() == dict(
        mean=0.0, q05=0.0, q50=0.0, q95=0.0
    )
    empty = cell_for(bundle.beta, "restart", "r1000")
    assert empty.exposures == empty.accepted == empty.refused == 0


def test_support_and_unavailable_incumbent():
    bundle = load_bundle()
    for request in [
        DecisionRequest(context=Context(segment="restart")),
        DecisionRequest(card_id="dec-2026", as_of="2026-12-09"),
        DecisionRequest(as_of="2026-12-09"),
        DecisionRequest(finance=Finance(principal_cents=3200000)),
        DecisionRequest(context=Context(business_age_years=59.0)),
    ]:
        result = evaluate(request, bundle)
        for choice in [result.greedy, result.cautious]:
            if choice.offer_id:
                assert next(
                    o for o in result.offers if o.offer_id == choice.offer_id
                ).support.automated
        if request.card_id == "dec-2026":
            assert result.incumbent_offer_id is None
            assert result.cautious.status == "review"
            assert result.cautious.immediate_review_cost_cents == 1500


def test_cautious_ranking_tie_and_numerical_boundary():
    rows = evaluate(DecisionRequest(), load_bundle()).offers

    def replace(index, lower, mean):
        item = rows[index]
        return item.model_copy(
            update={
                "improvement_cents": item.improvement_cents.model_copy(
                    update={"q05": lower}
                ),
                "expected_contribution_cents": item.expected_contribution_cents.model_copy(
                    update={"mean": mean}
                ),
            }
        )

    a, b, c = (
        replace(0, 3000.0, 90000.0),
        replace(1, 0.0, 70000.0),
        replace(2, 4000.0, 80000.0),
    )
    greedy, cautious = choose([a, b, c], "r900")
    assert greedy.offer_id == "r800" and cautious.offer_id == "r1000"
    assert choose([a, b, replace(2, 3000.0, 80000.0)], "r900")[1].offer_id == "r800"
    assert (
        choose([replace(0, 2001.0, 90000.0), b, replace(2, -1000.0, 80000.0)], "r900")[
            1
        ].reason
        == "numerically_borderline_improvement"
    )


def test_api_contract_and_deterministic_decision():
    client = TestClient(app)
    meta = client.get("/api/meta")
    assert (
        meta.status_code == 200 and meta.json()["provenance"]["simulation_only"] is True
    )
    r1 = client.post("/api/decision", json={}).json()
    r2 = client.post("/api/decision", json={}).json()
    r1.pop("elapsed_ms")
    r2.pop("elapsed_ms")
    assert r1 == r2
    assert client.post("/api/decision", json={"oracle": True}).status_code == 422
    assert client.post("/api/decision", json={"as_of": "2026-99-12"}).status_code == 422
