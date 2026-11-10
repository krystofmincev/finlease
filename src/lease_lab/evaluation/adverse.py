"""Predeclared challenges and point-only recalibration. Never imported by serving."""

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from lease_lab.artifacts.store import FrozenBundle
from lease_lab.contracts import Context, DecisionRequest
from lease_lab.data.events import Event
from lease_lab.decisions.policy import evaluate
from lease_lab.evaluation.metrics import prediction_score
from lease_lab.models.logistic_laplace import LogisticSampler, fit_logistic
from lease_lab.report_contracts import AuditFinding, PredictionScore
from lease_lab.simulation.environment import probability


def shifted_rows(
    rows: list[Event], seed: int, case_mix: bool = False, response_change: bool = False
) -> list[Event]:
    rng = np.random.default_rng(seed)
    results = []
    for event in rows:
        context = event.context
        if case_mix:
            context = context.model_copy(
                update={
                    "monthly_turnover_cents": round(
                        context.monthly_turnover_cents * 0.4
                    ),
                    "existing_debt_burden": min(
                        1.5, context.existing_debt_burden + 0.25
                    ),
                }
            )
        p = probability(context, int(event.offer_id[1:]), shifted=response_change)
        results.append(
            event.model_copy(
                update={"context": context, "accepted": bool(rng.random() < p)}
            )
        )
    return results


def adverse_checks(
    rows: list[Event], bundle: FrozenBundle
) -> tuple[list[AuditFinding], list[PredictionScore]]:
    findings = []
    scores = []
    fit = [e for e in rows if e.split == "fit"]
    try:
        fit_logistic([e.model_copy(update={"accepted": True}) for e in fit])
    except ValueError as error:
        findings.append(
            AuditFinding(
                name="Accepted-only input", status="rejected", detail=str(error)
            )
        )
    else:
        raise AssertionError("accepted-only model fit succeeded")
    for name, request in [
        (
            "Expired November card",
            DecisionRequest(method="laplace", as_of="2026-12-09"),
        ),
        (
            "New December card",
            DecisionRequest(method="laplace", card_id="dec-2026", as_of="2026-12-09"),
        ),
        (
            "Sparse restart cohort",
            DecisionRequest(
                method="laplace",
                context=Context(segment="restart", business_age_years=5.0),
            ),
        ),
    ]:
        result = evaluate(request, bundle)
        assert result.cautious.status == "review"
        assert all(not o.support.automated for o in result.offers)
        widths = ", ".join(
            f"{o.rate_bps / 100:g}%: {(o.acceptance.q95 - o.acceptance.q05) * 100:.1f}pp interval, {o.support.response_count} responses"
            for o in result.offers
        )
        findings.append(
            AuditFinding(
                name=name,
                status="review",
                detail=widths + ". Predictions do not override the evidence gate.",
            )
        )
    diagnostic = [e for e in rows if e.split == "diagnostic"]
    test = [e for e in rows if e.split == "test"]
    sampler = LogisticSampler(bundle.laplace["default"])

    def predict(cases):
        return sampler.predict([e.context for e in cases], [e.offer_id for e in cases])[
            0
        ]

    mixed = shifted_rows(test, 819, case_mix=True)
    scores.append(
        prediction_score(
            mixed,
            predict(mixed),
            "case-mix-shift",
            "laplace",
            "held-out challenge",
            "map-laplace-approximation",
            bundle=bundle,
        )
    )
    findings.append(
        AuditFinding(
            name="Input shift signal",
            status="input-drift",
            detail=f"Mean existing burden moves from {np.mean([e.context.existing_debt_burden for e in fit]):.3f} to {np.mean([e.context.existing_debt_burden for e in mixed]):.3f}. This is a visible-input signal, not proof of outcome degradation.",
        )
    )
    changed_diagnostic = shifted_rows(diagnostic, 820, response_change=True)
    changed_test = shifted_rows(test, 821, response_change=True)
    p_diag, p_test = predict(changed_diagnostic), predict(changed_test)
    scores.append(
        prediction_score(
            changed_test,
            p_test,
            "response-change",
            "laplace",
            "mature held-out challenge",
            "map-laplace-approximation",
            bundle=bundle,
        )
    )
    calibrator = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200).fit(
        logit(p_diag).reshape(-1, 1), [e.accepted for e in changed_diagnostic]
    )
    calibrated = calibrator.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]
    scores.append(
        prediction_score(
            changed_test,
            calibrated,
            "response-change",
            "point-platt-recalibrated",
            "mature held-out challenge",
            "not_estimated",
            bundle=bundle,
        )
    )
    findings.append(
        AuditFinding(
            name="Recalibration boundary",
            status="point-only",
            detail="A separate two-parameter Platt map uses 900 mature response-change diagnostic rows and is scored on 900 separate test rows. It is a point comparator; no calibration uncertainty is represented and no posterior draws were recalibrated. It cannot create experimental support for December offers or repair unavailable feedback.",
        )
    )
    return findings, scores
