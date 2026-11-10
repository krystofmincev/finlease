import json

import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import beta as beta_distribution

from lease_lab.artifacts.store import (
    digest,
    encode,
    load_bundle,
    load_report,
    load_replays,
    source_digest,
)
from lease_lab.config import ROOT
from lease_lab.report_contracts import (
    SelectionCheckpoint,
    require_completed_comparison,
    selection_payload,
)
from lease_lab.contracts import DecisionRequest
from lease_lab.decisions.policy import evaluate
from lease_lab.models.beta_cells import cell_for


def test_saved_split_identity_denominators_and_support():
    bundle = load_bundle()
    report = load_report(bundle)
    assert bundle.provenance.source_sha256 == source_digest(), (
        "rebuild offline after source changes"
    )
    ids = json.loads(
        (
            ROOT / "artifacts/runs" / bundle.provenance.run_id / "split-ids.json"
        ).read_text()
    )["ids"]
    flat = [i for rows in ids.values() for i in rows]
    assert len(set(flat)) == len(flat) == 6000
    assert report.datasets[0].split_ids_sha256 == digest(encode(ids))
    require_completed_comparison(report)
    selection = SelectionCheckpoint.model_validate_json(
        (
            ROOT / "artifacts/runs" / bundle.provenance.run_id / "selection.json"
        ).read_bytes()
    )
    assert (
        digest(encode(selection_payload(selection))) == report.comparison.freeze_sha256
    )
    assert selection.pre_generation_sha256 is not None
    assert all(
        i.startswith(report.comparison.final_namespace + "-") for i in ids["test"]
    )
    for roles in selection.roles.values():
        assert set(roles["early_stop"]).isdisjoint(roles["report_only"])
        assert set(roles["early_stop"] + roles["report_only"]) == set(
            roles["diagnostic"]
        )
        assert (
            set(roles["fit_512"])
            <= set(roles["fit_1800"])
            <= set(roles["fit_3600"])
            == set(roles["fit"])
        )
        assert set(ids["test"]).isdisjoint(
            i for values in roles.values() for i in values
        )
    for score in report.prediction:
        assert score.advisory_supported_n + score.advisory_unsupported_n == score.n
        assert score.n + score.missing_prediction_n == score.cohort_n
        assert sum(b.count for b in score.calibration) == score.n
    for score in report.policies:
        assert score.n == 900
        assert score.review_handling_cost_cents == score.reviews * 1500
        assert score.recommendation_coverage == pytest.approx(
            (score.n - score.reviews) / score.n
        )
        assert score.support_violations == 0


def test_exact_joint_improvement_quantile_against_independent_integral():
    bundle = load_bundle()
    # D = 170000 p(10%) - 125000 p(9%); independent Beta cell posteriors.
    incumbent = cell_for(bundle.beta, "established", "r900")
    candidate = cell_for(bundle.beta, "established", "r1000")
    a0, b0 = incumbent.accepted + 1, incumbent.refused + 1
    a1, b1 = candidate.accepted + 1, candidate.refused + 1

    def cdf(delta):
        return quad(
            lambda p0: beta_distribution.pdf(p0, a0, b0)
            * beta_distribution.cdf((delta + 125000 * p0) / 170000, a1, b1),
            0,
            1,
            epsabs=1e-10,
        )[0]

    reference = brentq(lambda d: cdf(d) - 0.05, -125000, 170000, xtol=1e-5)
    for n in (4096, 16384):
        result = evaluate(DecisionRequest(draw_count=n), bundle)
        q = next(
            o.improvement_cents.q05 for o in result.offers if o.offer_id == "r1000"
        )
        assert (
            abs(q - reference) < 500
        )  # declared €5 fixture MC tolerance; no global precision guarantee


def test_presets_match_ui_steps_and_replays_remain_factual():
    bundle = load_bundle()
    for context in bundle.case_presets:
        assert context.business_age_years * 2 == round(context.business_age_years * 2)
        assert context.monthly_turnover_cents % 100000 == 0
        assert context.existing_debt_burden == round(context.existing_debt_burden, 2)
    report = load_replays(bundle)
    assert len(report.summaries) == 24 and all(s.seeds == 20 for s in report.summaries)
    assert len(report.paths) == 72
    for path in report.paths:
        chosen = {f.index: f.chosen_offer_id for f in path.frames}
        for f in path.frames:
            assert f.matured + f.pending == report.initial_evidence_count + f.index + 1
            for feedback in f.feedback:
                assert feedback.offer_id == chosen[feedback.decision_index]
                assert feedback.available_at <= f.decision_at
