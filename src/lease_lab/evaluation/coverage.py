"""Fixed-world repeated-data coverage, with model-specific targets. Not SBC."""

import numpy as np

from lease_lab.config import CONFIG, HISTORICAL_OFFERS
from lease_lab.contracts import Context
from lease_lab.evaluation.metrics import wilson
from lease_lab.models.beta_cells import beta_summary, cell_for, fit_beta
from lease_lab.models.logistic_laplace import LogisticSampler, fit_logistic
from lease_lab.report_contracts import CoverageRow
from lease_lab.simulation.environment import contexts, generate, probability


def coverage_checks() -> list[CoverageRow]:
    output = []
    probes = [
        Context(
            segment="established",
            business_age_years=12.0,
            monthly_turnover_cents=3000000,
            existing_debt_burden=0.18,
        ),
        Context(
            segment="new",
            business_age_years=1.5,
            monthly_turnover_cents=1800000,
            existing_debt_burden=0.25,
        ),
        Context(
            segment="restart",
            business_age_years=5.0,
            monthly_turnover_cents=2500000,
            existing_debt_burden=0.18,
        ),
    ]
    for family in ("smooth", "nonlinear"):
        targets = {}
        target_error = {}
        for segment in ("established", "new", "restart"):
            # Independent Monte Carlo integration of the declared cohort context law.
            cases = contexts(20000, np.random.default_rng(7001), segment, family=family)
            for offer_id in HISTORICAL_OFFERS:
                probabilities = np.array(
                    [probability(c, int(offer_id[1:]), family) for c in cases]
                )
                targets[(segment, offer_id)] = float(probabilities.mean())
                target_error[(segment, offer_id)] = float(
                    probabilities.std(ddof=1) / np.sqrt(len(cases))
                )
        values = {
            (method, i, o): []
            for method in ("beta", "laplace")
            for i in range(len(probes))
            for o in HISTORICAL_OFFERS
        }
        widths = {key: [] for key in values}
        counts = {key: [] for key in values}
        for seed in range(CONFIG["coverage_seeds"]):
            fit = [
                e
                for e in generate(
                    CONFIG["coverage_applicants"],
                    9000 + seed,
                    family,
                    namespace=f"coverage-{seed}",
                )
                if e.split == "fit"
            ]
            beta = fit_beta(fit)
            sampler = LogisticSampler(fit_logistic(fit))
            for i, probe in enumerate(probes):
                _, low, high = sampler.predict([probe] * 3, HISTORICAL_OFFERS)
                for j, offer_id in enumerate(HISTORICAL_OFFERS):
                    cell = cell_for(beta, probe.segment, offer_id)
                    exact = beta_summary(cell.accepted + 1, cell.refused + 1)
                    for method, l, h, target in [
                        (
                            "beta",
                            exact.q05,
                            exact.q95,
                            targets[(probe.segment, offer_id)],
                        ),
                        (
                            "laplace",
                            low[j],
                            high[j],
                            probability(probe, int(offer_id[1:]), family),
                        ),
                    ]:
                        key = (method, i, offer_id)
                        values[key].append(bool(l <= target <= h))
                        widths[key].append(float(h - l))
                        counts[key].append(cell.accepted + cell.refused)
        for (method, i, offer_id), covered in values.items():
            n = len(covered)
            success = sum(covered)
            low, high = wilson(success, n)
            segment = probes[i].segment
            support = (
                "zero"
                if max(counts[(method, i, offer_id)]) == 0
                else "sparse"
                if np.mean(counts[(method, i, offer_id)]) < 30
                else "dense"
            )
            output.append(
                CoverageRow(
                    family=family,
                    model=method,
                    probe=f"{segment} · {int(offer_id[1:]) / 100:g}%",
                    target="cohort-average probability"
                    if method == "beta"
                    else "full-visible-context probability",
                    support=support,
                    seeds=n,
                    covered=success,
                    coverage=success / n,
                    interval_low=low,
                    interval_high=high,
                    mean_width=float(np.mean(widths[(method, i, offer_id)])),
                    target_standard_error=target_error[(segment, offer_id)]
                    if method == "beta"
                    else 0.0,
                )
            )
    return output
