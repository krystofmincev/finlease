"""OFFLINE ONLY: declared visible-context response mechanisms, never learner input."""

from functools import lru_cache
import numpy as np
from scipy.special import expit

from lease_lab.config import CONFIG, SEGMENTS
from lease_lab.contracts import Context, Finance
from lease_lab.data.events import Event
from lease_lab.domain.finance import payment

SCENARIO_VERSIONS = {
    "smooth": "smooth-logistic-v1",
    "nonlinear": "complex-economic-mixture-v1",
}


def contexts(
    n: int,
    rng: np.random.Generator,
    forced_segment: str | None = None,
    *,
    family: str = "smooth",
) -> list[Context]:
    segments = rng.choice(SEGMENTS, n, p=CONFIG["segment_probabilities"])
    ages = {
        "new": (0.2, 3),
        "growing": (3, 8),
        "established": (8, 25),
        "restart": (0.5, 20),
    }
    result = []
    for segment in segments:
        segment = forced_segment or str(segment)
        age = rng.uniform(*ages[segment])
        if family == "nonlinear":
            z_scale, z_turnover, z_debt = rng.normal(size=3)
            stage = SEGMENTS.index(segment)
            log_turnover = (
                np.log(25000)
                + (-0.35, 0.0, 0.30, -0.20)[stage]
                + 0.18 * np.log((age + 1) / 6)
                + 0.55 * z_scale
                + 0.25 * z_turnover
            )
            turnover = np.clip(np.exp(log_turnover), 5000, 150000)
            burden = float(
                0.85
                * expit(
                    -1.55
                    + (0.15, 0.05, -0.15, 0.25)[stage]
                    - 0.35 * z_scale
                    + 0.65 * z_debt
                )
            )
        elif family == "smooth":
            turnover = np.clip(rng.lognormal(np.log(25000), 0.65), 5000, 150000)
            burden = float(rng.beta(2, 8) * 0.85)
        else:
            raise ValueError("unknown simulator family")
        result.append(
            Context(
                segment=segment,
                business_age_years=float(age),
                monthly_turnover_cents=round(turnover * 100),
                existing_debt_burden=burden,
            )
        )
    return result


@lru_cache(maxsize=128)
def _payment_difference(
    rate_bps: int, principal: int, term: int, balloon: int
) -> float:
    finance = Finance(
        principal_cents=principal, term_months=term, balloon_cents=balloon
    )
    return (payment(finance, rate_bps)[0] - payment(finance, 900)[0]) / 1000


def complex_components(
    context: Context, rate_bps: int, shifted: bool = False
) -> list[tuple[float, float]]:
    """Simulation-only (weight, response) enumeration; no sampled personal truth."""
    L = np.log(context.monthly_turnover_cents / 2500000)
    b = (context.existing_debt_burden - 0.18) / 0.15
    q_urgency = expit(-0.8 + 0.5 * (context.segment in ("new", "growing")) + 0.6 * b)
    base = (
        0.8
        + 0.2 * (context.business_age_years - 8) / 10
        + 0.3 * L
        - 0.65 * b
        + dict(zip(SEGMENTS, (-0.45, 0.0, 0.35, -0.70)))[context.segment]
    )
    d = _payment_difference(
        rate_bps,
        CONFIG["principal_cents"],
        CONFIG["term_months"],
        CONFIG["balloon_cents"],
    )
    shift = -0.9 - 0.95 * ((rate_bps - 900) / 100) if shifted else 0.0
    components = []
    for urgency in (0, 1):
        q_other = expit(
            -0.25 + 0.5 * L + 0.3 * (context.segment == "established") - 0.9 * urgency
        )
        for other in (0, 1):
            weight = (q_urgency if urgency else 1 - q_urgency) * (
                q_other if other else 1 - q_other
            )
            beta = (0.6 + 0.4 * np.clip(context.existing_debt_burden / 0.3, 0, 2)) * (
                1 - 0.35 * urgency
            )
            p = expit(
                base
                + 0.85 * urgency
                - other
                - beta * d
                - 0.8 * other * max(d, 0)
                + shift
            )
            components.append((float(weight), float(p)))
    return components


def probability(
    context: Context, rate_bps: int, family: str = "smooth", shifted: bool = False
) -> float:
    if family == "nonlinear":
        return sum(
            weight * p for weight, p in complex_components(context, rate_bps, shifted)
        )
    if family != "smooth":
        raise ValueError("unknown simulator family")
    log_turnover = np.log(context.monthly_turnover_cents / 2000000)
    burden = (context.existing_debt_burden - 0.18) / 0.15
    rate = (rate_bps - 900) / 100
    segment_effect = {
        "new": -0.45,
        "growing": 0.0,
        "established": 0.35,
        "restart": -0.7,
    }[context.segment]
    score = (
        1.05
        + 0.25 * (context.business_age_years - 8) / 10
        + 0.45 * log_turnover
        - 0.8 * burden
        - 0.9 * rate
        + segment_effect
    )
    if shifted:
        score += (
            -0.9 - 0.95 * rate
        )  # predeclared stronger rate sensitivity after change
    return float(expit(score))


def generate(
    n: int = 6000,
    seed: int = 20261109,
    family: str = "smooth",
    namespace: str = "offline",
) -> list[Event]:
    split_rng, context_rng, assignment_rng, response_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    # Partition applicant IDs before generating contexts or outcomes.
    counts = [
        int(n * CONFIG[name])
        for name in ("fit_fraction", "diagnostic_fraction", "selection_fraction")
    ]
    split_names = np.array(
        ["fit"] * counts[0]
        + ["diagnostic"] * counts[1]
        + ["selection"] * counts[2]
        + ["test"] * (n - sum(counts))
    )
    split_rng.shuffle(split_names)
    cases = contexts(n, context_rng, family=family)
    result = []
    for i, context in enumerate(cases):
        # Deliberate historical exposure design; no outcome-dependent deletion.
        allowed = (
            ["r800", "r900"]
            if context.segment == "restart"
            else ["r800", "r900", "r1000"]
        )
        chosen = str(assignment_rng.choice(allowed))
        accepted = bool(
            response_rng.random() < probability(context, int(chosen[1:]), family)
        )
        result.append(
            Event(
                applicant_id=f"{namespace}-{i:06}",
                split=str(split_names[i]),
                context=context,
                offer_id=chosen,
                assignment_probability=1.0 / len(allowed),
                decision_at=i,
                outcome_at=i + 1,
                available_at=i + 1,
                accepted=accepted,
            )
        )
    return result
