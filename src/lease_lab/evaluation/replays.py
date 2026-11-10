"""Offline matched-stream evaluation. Environment truth is never passed to choices."""

from dataclasses import dataclass
import time

import numpy as np

from lease_lab.config import CONFIG, HISTORICAL_OFFERS
from lease_lab.contracts import Provenance
from lease_lab.data.events import Event
from lease_lab.models.online import OnlineLearner
from lease_lab.report_contracts import (
    Feedback,
    ReplayFrame,
    ReplayPath,
    ReplayReport,
    ReplaySummary,
)
from lease_lab.simulation.environment import contexts, generate, probability

SCENARIOS = ("stationary", "delayed", "changed-response", "outcome-dependent-delay")


@dataclass(frozen=True)
class ReplayOutcome:
    path: ReplayPath
    realised: float
    expected: float
    regret: float


def replay(
    seed: int,
    method: str,
    policy: str,
    scenario: str,
    n: int = 400,
    record: bool = True,
) -> ReplayOutcome:
    if scenario not in SCENARIOS:
        raise ValueError("unknown replay scenario")
    streams = np.random.SeedSequence(60000 + seed).spawn(4)
    context_rng, response_rng, delay_rng, choice_rng = [
        np.random.default_rng(s) for s in streams
    ]
    cases = contexts(n, context_rng)
    uniforms = response_rng.random((n, 3))
    delays = delay_rng.integers(0, 31 if scenario == "delayed" else 7, (n, 3))
    initial = [
        e.model_copy(
            update={
                "split": "replay",
                "decision_at": 0,
                "outcome_at": 0,
                "available_at": 0,
            }
        )
        for e in generate(
            CONFIG["replay_initial_applicants"],
            50000 + seed,
            namespace=f"initial-{seed}",
        )
    ]
    learner = OnlineLearner(initial, method)
    queue = []
    frames = []
    realised = 0.0
    expected = 0.0
    regret = 0.0
    width = learner.width()
    width_at = 0
    margins = np.array(
        [CONFIG["acceptance_contribution_cents"][o] for o in HISTORICAL_OFFERS]
    )
    for index, context in enumerate(cases):
        clock = index + 1
        matured = [e for e in queue if e.available_at <= clock]
        queue = [e for e in queue if e.available_at > clock]
        feedback = []
        for event in matured:
            learner.observe(event, clock)
            if event.accepted:
                realised += CONFIG["acceptance_contribution_cents"][event.offer_id]
            feedback.append(
                Feedback(
                    decision_index=event.decision_at - 1,
                    offer_id=event.offer_id,
                    outcome_at=event.outcome_at,
                    available_at=event.available_at,
                    accepted=event.accepted,
                )
            )
        chosen = learner.choose(
            context, policy, int(choice_rng.integers(0, 2147483647))
        )
        j = HISTORICAL_OFFERS.index(chosen)
        shifted = scenario == "changed-response" and index >= n // 2
        probabilities = np.array(
            [
                probability(context, int(o[1:]), shifted=shifted)
                for o in HISTORICAL_OFFERS
            ]
        )
        accepted = bool(uniforms[index, j] < probabilities[j])
        delay = int(delays[index, j])
        if scenario == "outcome-dependent-delay":
            delay = 40 if accepted else 0
        occurrence = clock + 1
        available = occurrence + delay
        factual = Event(
            applicant_id=f"replay-{seed}-{index}",
            split="replay",
            context=context,
            offer_id=chosen,
            assignment_probability=None,
            decision_at=clock,
            outcome_at=occurrence,
            available_at=available,
            accepted=accepted,
        )
        learner.assign(factual.model_copy(update={"accepted": None}))
        queue.append(factual)
        realised -= CONFIG["decision_cost_cents"]
        expected += float(probabilities[j] * margins[j] - CONFIG["decision_cost_cents"])
        regret += float(np.max(probabilities * margins) - probabilities[j] * margins[j])
        if index % 10 == 0 or index == n - 1:
            width = learner.width()
            width_at = index
        if record:
            exposure, responses = learner.counts()
            frames.append(
                ReplayFrame(
                    index=index,
                    chosen_offer_id=chosen,
                    decision_at=clock,
                    outcome_at=occurrence,
                    available_at=available,
                    matured=len(learner.history),
                    pending=len(queue),
                    exposures=exposure,
                    responses=responses,
                    mean_interval_width=width,
                    interval_as_of_index=width_at,
                    cumulative_realised_cents=realised,
                    cumulative_expected_cents=expected,
                    cumulative_oracle_regret_cents=regret,
                    feedback=feedback,
                )
            )
    path = ReplayPath(
        model=method, policy=policy, scenario=scenario, seed=seed, frames=frames
    )
    return ReplayOutcome(path, realised, expected, regret)


def build_replays(provenance: Provenance) -> ReplayReport:
    started = time.perf_counter()
    paths = []
    summaries = []
    for scenario in SCENARIOS:
        for method in ("beta", "laplace"):
            runs = {
                policy: [
                    replay(
                        seed,
                        method,
                        policy,
                        scenario,
                        CONFIG["replay_decisions"],
                        record=seed < 3,
                    )
                    for seed in range(CONFIG["replay_seeds"])
                ]
                for policy in ("fixed", "greedy", "thompson")
            }
            fixed = np.array([r.expected for r in runs["fixed"]])
            for policy, results in runs.items():
                expected = np.array([r.expected for r in results])
                difference = expected - fixed
                paths.extend(r.path for r in results if r.path.frames)
                low, high = np.quantile(expected, [0.05, 0.95])
                # Across-seed paired differences: retain seed variability; this is
                # not a posterior or an exact confidence interval on the mean.
                dlow, dhigh = np.quantile(difference, [0.05, 0.95])
                summaries.append(
                    ReplaySummary(
                        model=method,
                        policy=policy,
                        scenario=scenario,
                        seeds=len(results),
                        mean_realised_cents=float(
                            np.mean([r.realised for r in results])
                        ),
                        mean_expected_cents=float(expected.mean()),
                        expected_q05_cents=float(low),
                        expected_q95_cents=float(high),
                        mean_regret_cents=float(np.mean([r.regret for r in results])),
                        paired_vs_fixed_mean_cents=float(difference.mean()),
                        paired_vs_fixed_low_cents=float(dlow),
                        paired_vs_fixed_high_cents=float(dhigh),
                    )
                )
    return ReplayReport(
        provenance=provenance,
        paths=paths,
        summaries=summaries,
        scenarios=list(SCENARIOS),
        seed_ids=[0, 1, 2],
        initial_evidence_count=CONFIG["replay_initial_applicants"],
        decisions_per_seed=CONFIG["replay_decisions"],
        elapsed_seconds=time.perf_counter() - started,
        interpretation=[
            "Authorised simulator exploration only; all policies share initial evidence, menus, contexts, outcome uniforms and delay mechanisms.",
            "Only the chosen matured response updates a learner. Occurrence and availability differ. Pending is never refusal.",
            "All 20 seeds contribute summaries. Detailed, predeclared seed paths 0, 1 and 2 are saved; none was selected for a winner.",
            "Realised contribution credits income only on feedback arrival and charges costs at decision; pending income is not credited. Expected contribution is separately evaluator-known.",
            "Regret uses the best eligible visible-context expected reward; no hidden applicant traits or sampled future outcomes.",
            "Across-seed 5–95% ranges are variability, not posterior intervals or confidence intervals on a mean. Thompson propensities are not computed.",
            "Changed response mechanism and outcome-dependent delay are failure scenarios; stationary Bayes and Thompson do not solve nonstationarity or selective availability.",
            "Mean interval width is for a fixed established probe across the three rates, refreshed every 10 decisions; pooled fits refresh after 25 new matured responses.",
        ],
    )
