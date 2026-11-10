from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from lease_lab.contracts import Context, DecisionRequest, Finance
from lease_lab.artifacts.store import Range, load_bundle
from lease_lab.config import CONFIG
from lease_lab.data.events import Event, require_binary_outcomes
from lease_lab.domain.finance import payment, payment_schedule
from lease_lab.domain.menus import menu
from lease_lab.simulation.environment import generate


def reference_payment(principal, months, balloon, bps):
    with localcontext() as ctx:
        ctx.prec = 50
        p, b = Decimal(principal), Decimal(balloon)
        r = Decimal(bps) / Decimal(120000)
        amount = (
            (p - b) / months
            if not r
            else (p - b / (1 + r) ** months) * r / (1 - (1 + r) ** (-months))
        )
        return int(amount.quantize(Decimal(1), rounding="ROUND_HALF_UP"))


@pytest.mark.parametrize(
    "principal,months,balloon,bps",
    [
        (3100000, 60, 300000, 800),
        (3100000, 60, 300000, 900),
        (3100000, 60, 300000, 1000),
        (120000, 12, 0, 0),
        (3100000, 60, 300000, 0),
        (1, 1, 0, 1),
        (100000000, 120, 0, 10000),
    ],
)
def test_payment_independent_decimal(principal, months, balloon, bps):
    f = Finance(principal_cents=principal, term_months=months, balloon_cents=balloon)
    monthly, total, interest = payment(f, bps)
    assert monthly == reference_payment(principal, months, balloon, bps)
    schedule = payment_schedule(f, bps)
    assert total == (months - 1) * monthly + schedule.last_instalment_cents + balloon
    assert total - principal == interest
    with localcontext() as ctx:
        ctx.prec = 50
        rate = Decimal(bps) / Decimal(120000)
        present = (
            sum(Decimal(monthly) / (1 + rate) ** m for m in range(1, months))
            + (Decimal(schedule.last_instalment_cents) + Decimal(balloon))
            / (1 + rate) ** months
        )
        assert abs(present - Decimal(principal)) <= Decimal(".50001")
    if bps == 0:
        assert interest == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"principal_cents": 0},
        {"term_months": 0},
        {"balloon_cents": 4000000},
        {"principal_cents": True},
    ],
)
def test_invalid_finance(bad):
    with pytest.raises(ValidationError):
        Finance(**bad)


def test_split_randomization_and_classes():
    rows = generate(600, 111)
    ids = [e.applicant_id for e in rows]
    assert len(set(ids)) == 600
    assert {
        name: sum(e.split == name for e in rows)
        for name in ("fit", "diagnostic", "selection", "test")
    } == {"fit": 360, "diagnostic": 90, "selection": 60, "test": 90}
    require_binary_outcomes([e for e in rows if e.split == "fit"])
    for event in rows:
        assert event.assignment_probability == (
            0.5 if event.context.segment == "restart" else 1 / 3
        )
        assert not (event.context.segment == "restart" and event.offer_id == "r1000")
    assert [e.model_dump() for e in rows] == [
        e.model_dump() for e in generate(600, 111)
    ]


def test_hidden_nested_fields_and_nonfinite_rejected():
    for bad in (
        {"oracle_probability": 0.9},
        {"context": {"true_acceptance": 0.8}},
        {"context": {"existing_debt_burden": float("nan")}},
    ):
        with pytest.raises(ValidationError):
            DecisionRequest.model_validate(bad)
    event = generate(1, 20)[0].model_dump()
    event["unchosen_outcomes"] = [True, False]
    with pytest.raises(ValidationError):
        Event.model_validate(event)


def test_accepted_only_rejected_and_card_expiry():
    rows = generate(60, 5)
    with pytest.raises(ValueError, match="acceptance_not_estimable"):
        require_binary_outcomes([e.model_copy(update={"accepted": True}) for e in rows])
    assert menu("nov-2026", "2026-11-09")[1]
    assert not menu("nov-2026", "2026-12-01")[1]
    assert {a.offer_id for a in menu("dec-2026", "2026-12-01")[0]} == {
        "r850",
        "r950",
        "r1050",
    }


def test_exact_half_cent_interest_only_and_configured_splits(monkeypatch):
    assert (
        payment(Finance(principal_cents=3000060, balloon_cents=3000060), 1000)[0]
        == 25001
    )
    assert (
        payment(
            Finance(principal_cents=3000075, balloon_cents=3000075, term_months=24), 800
        )[0]
        == 20001
    )
    monkeypatch.setitem(CONFIG, "fit_fraction", 0.5)
    monkeypatch.setitem(CONFIG, "diagnostic_fraction", 0.2)
    monkeypatch.setitem(CONFIG, "selection_fraction", 0.2)
    rows = generate(100, 33)
    assert [
        sum(e.split == name for e in rows)
        for name in ("fit", "diagnostic", "selection", "test")
    ] == [50, 20, 20, 10]
    with pytest.raises(ValueError, match="overpay"):
        payment(Finance(principal_cents=60, balloon_cents=0, term_months=120), 0)


def test_corrupt_frozen_counts_ranges_dates_and_provenance():
    bundle = load_bundle()
    cell = bundle.beta.cells[0].model_dump()
    for bad in [{"accepted": -1}, {"exposures": 0}]:
        with pytest.raises(ValidationError):
            type(bundle.beta.cells[0]).model_validate(cell | bad)
    with pytest.raises(ValidationError):
        Range(minimum=2.0, maximum=1.0)
    with pytest.raises(ValidationError):
        type(bundle.provenance).model_validate(
            bundle.provenance.model_dump() | {"data_sha256": "arbitrary"}
        )
    with pytest.raises(ValidationError):
        DecisionRequest(as_of="2026-02-31")
    with pytest.raises(ValidationError):
        type(bundle.support).model_validate(
            bundle.support.model_dump() | {"by_segment": {}}
        )


def test_complex_revision_preserves_smooth_bytes_support_and_probabilities():
    import json
    import numpy as np
    from lease_lab.artifacts.store import digest, encode
    from lease_lab.builds.exact import support_profile
    from lease_lab.config import ROOT
    from lease_lab.data.features import fit_transform
    from lease_lab.simulation.environment import probability

    before = json.loads((ROOT / "tests/fixtures/smooth-v1-reference.json").read_bytes())
    rows = generate(6000, 20261109)
    assert (
        digest(encode([e.model_dump(mode="json") for e in rows]))
        == before["data_sha256"]
    )
    for item in before["probabilities"]:
        assert (
            probability(
                Context.model_validate(item["context"]),
                item["rate"],
                shifted=item["shifted"],
            )
            == item["p"]
        )
    historical = json.loads(
        (ROOT / "artifacts/runs/core-01e96c4ee2b1/bundle.json").read_bytes()
    )
    fit = [e for e in rows if e.split == "fit"]
    assert support_profile(fit).model_dump(mode="json") == historical["support"]
    assert (
        fit_transform(fit).model_dump(mode="json")
        == historical["laplace"]["default"]["transform"]
    )
    complex_rows = generate(6000, 20261109, "nonlinear")
    assert complex_rows == generate(6000, 20261109, "nonlinear")
    for a, b in zip(rows, complex_rows):
        assert (
            a.applicant_id,
            a.split,
            a.context.segment,
            a.offer_id,
            a.assignment_probability,
            a.decision_at,
            a.outcome_at,
            a.available_at,
        ) == (
            b.applicant_id,
            b.split,
            b.context.segment,
            b.offer_id,
            b.assignment_probability,
            b.decision_at,
            b.outcome_at,
            b.available_at,
        )
    complex_fit = [e for e in complex_rows if e.split == "fit"]
    assert {e.accepted for e in complex_fit} == {True, False}
    assert support_profile(complex_fit) != support_profile(fit)
    assert fit_transform(complex_fit) != fit_transform(fit)
    for segment in ("new", "growing", "established", "restart"):
        from lease_lab.simulation.environment import contexts

        cases = contexts(2000, np.random.default_rng(7001), segment, family="nonlinear")
        relation = np.corrcoef(
            np.log([c.monthly_turnover_cents for c in cases]),
            [c.existing_debt_burden for c in cases],
        )[0, 1]
        assert relation < 0  # declared development-only attribute hypothesis
    with pytest.raises(ValueError):
        support_profile([e for e in rows if e.split == "test"])
    for hidden in ("urgency", "competing_offer", "latent_scale"):
        with pytest.raises(ValidationError):
            Event.model_validate(complex_rows[0].model_dump() | {hidden: True})


def independent_complex(context, rate, shifted=False):
    import math

    sigmoid = lambda z: 1 / (1 + math.exp(-z))
    L = math.log(context.monthly_turnover_cents / 2500000)
    b = (context.existing_debt_burden - 0.18) / 0.15
    qu = sigmoid(-0.8 + 0.5 * (context.segment in ("new", "growing")) + 0.6 * b)
    d = (
        reference_payment(3100000, 60, 300000, rate)
        - reference_payment(3100000, 60, 300000, 900)
    ) / 1000
    base = (
        0.8
        + 0.2 * (context.business_age_years - 8) / 10
        + 0.3 * L
        - 0.65 * b
        + {"new": -0.45, "growing": 0.0, "established": 0.35, "restart": -0.7}[
            context.segment
        ]
    )
    result = []
    for u in (0, 1):
        qo = sigmoid(
            -0.25 + 0.5 * L + 0.3 * (context.segment == "established") - 0.9 * u
        )
        beta = (0.6 + 0.4 * min(2, max(0, context.existing_debt_burden / 0.3))) * (
            1 - 0.35 * u
        )
        for o in (0, 1):
            weight = (qu if u else 1 - qu) * (qo if o else 1 - qo)
            shift = -0.9 - 0.95 * ((rate - 900) / 100) if shifted else 0
            p = sigmoid(base + 0.85 * u - o - beta * d - 0.8 * o * max(d, 0) + shift)
            result.append((weight, p))
    return result, qu


@pytest.mark.parametrize(
    "context",
    [
        Context(),
        Context(
            segment="new",
            business_age_years=0.2,
            monthly_turnover_cents=500000,
            existing_debt_burden=0.85,
        ),
        Context(
            segment="established",
            business_age_years=25.0,
            monthly_turnover_cents=15000000,
            existing_debt_burden=0.01,
        ),
        Context(
            segment="restart",
            business_age_years=60.0,
            monthly_turnover_cents=10000,
            existing_debt_burden=2.0,
        ),
    ],
)
def test_exact_complex_marginal_directions_and_latent_sampling(context):
    import numpy as np
    from lease_lab.simulation.environment import complex_components, probability

    for shifted in (False, True):
        values = []
        for rate in (800, 900, 1000):
            expected, qu = independent_complex(context, rate, shifted)
            actual = complex_components(context, rate, shifted)
            np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
            assert sum(w for w, p in actual) == pytest.approx(1.0, abs=2e-15)
            marginal = sum(w * p for w, p in expected)
            assert probability(context, rate, "nonlinear", shifted) == pytest.approx(
                marginal, abs=2e-15
            )
            assert 0 <= marginal <= 1
            assert actual[1][1] < actual[0][1] and actual[3][1] < actual[2][1]
            values.append(marginal)
        assert values[0] >= values[1] >= values[2]
    # Explicit U/O/Y sampling, independent from the collapsed production generator.
    expected, qu = independent_complex(context, 900)
    rng = np.random.default_rng(881)
    n = 100000
    u = rng.random(n) < qu
    q_other = np.array([expected[1][0] / (1 - qu), expected[3][0] / qu])
    o = rng.random(n) < q_other[u.astype(int)]
    component_p = np.array([p for w, p in expected])
    sampled = rng.random(n) < component_p[2 * u.astype(int) + o.astype(int)]
    marginal = sum(w * p for w, p in expected)
    assert (
        abs(sampled.mean() - marginal)
        <= 5 * np.sqrt(marginal * (1 - marginal) / n) + 1 / n
    )
    assert any(abs(p - marginal) > 1e-4 for w, p in expected)


def test_coverage_samples_the_family_context_law(monkeypatch):
    from lease_lab.evaluation import coverage
    from lease_lab.simulation.environment import contexts

    seen = []

    def observed(n, rng, segment, *, family):
        assert n == 20000
        seen.append((family, segment))
        return contexts(24, rng, segment, family=family)

    monkeypatch.setattr(coverage, "contexts", observed)
    monkeypatch.setitem(CONFIG, "coverage_seeds", 1)
    monkeypatch.setitem(CONFIG, "coverage_applicants", 200)
    result = coverage.coverage_checks()
    assert seen == [
        (family, segment)
        for family in ("smooth", "nonlinear")
        for segment in ("established", "new", "restart")
    ]
    assert len(result) == 36
