import pytest

from lease_lab.contracts import Context
from lease_lab.data.events import Event
from lease_lab.evaluation.replays import replay
from lease_lab.models.online import OnlineLearner
from lease_lab.simulation.environment import generate


def initial_rows():
    return [
        e.model_copy(
            update={
                "split": "replay",
                "decision_at": 0,
                "outcome_at": 0,
                "available_at": 0,
            }
        )
        for e in generate(90, 50000)
    ]


def test_only_chosen_available_feedback_updates_and_no_double_update():
    learner = OnlineLearner(initial_rows(), "beta")
    event = Event(
        applicant_id="chosen-1",
        split="replay",
        context=Context(),
        offer_id="r900",
        assignment_probability=None,
        decision_at=1,
        outcome_at=2,
        available_at=8,
        accepted=None,
    )
    learner.assign(event)
    after_assign = [c.model_dump() for c in learner.beta.cells]
    assert sum(c.accepted + c.refused for c in learner.beta.cells) == 90
    assert sum(c.pending for c in learner.beta.cells) == 1
    for bad in [
        event,
        event.model_copy(update={"accepted": False}),
        event.model_copy(update={"offer_id": "r1000", "accepted": True}),
    ]:
        with pytest.raises(ValueError):
            learner.observe(bad, clock=3)
    assert [c.model_dump() for c in learner.beta.cells] == after_assign
    matured = event.model_copy(update={"accepted": True})
    learner.observe(matured, clock=8)
    assert sum(c.accepted + c.refused for c in learner.beta.cells) == 91
    assert sum(c.pending for c in learner.beta.cells) == 0
    with pytest.raises(ValueError):
        learner.observe(matured, clock=9)
    with pytest.raises(ValueError):
        learner.observe(
            matured.model_copy(update={"applicant_id": "never-chosen"}), clock=9
        )


@pytest.mark.parametrize("method", ["beta", "laplace"])
def test_replay_is_repeatable_and_available_feedback_is_factual(method):
    first = replay(0, method, "thompson", "delayed", n=45)
    second = replay(0, method, "thompson", "delayed", n=45)
    assert first.path.model_dump() == second.path.model_dump()
    assert first.realised == second.realised
    assert first.expected == second.expected
    assert any(f.pending > 3 for f in first.path.frames)
    chosen = {f.index: f.chosen_offer_id for f in first.path.frames}
    for frame in first.path.frames:
        assert frame.matured + frame.pending == 90 + frame.index + 1
        assert sum(frame.exposures) == 90 + frame.index + 1
        assert sum(frame.responses) == frame.matured
        assert frame.assignment_probability is None
        for feedback in frame.feedback:
            assert feedback.available_at <= frame.decision_at
            assert feedback.offer_id == chosen[feedback.decision_index]


def test_observed_delayed_fixed_fit_plateau_is_refined_without_weakening_gate(
    monkeypatch,
):
    from lease_lab.models import online

    original = online.fit_logistic
    diagnostics = []

    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        diagnostics.append(result.diagnostics)
        return result

    monkeypatch.setattr(online, "fit_logistic", checked)
    replay(0, "laplace", "fixed", "delayed", n=400, record=False)
    refined = [d for d in diagnostics if d.newton_refinement_steps]
    assert refined
    assert all(d.converged and d.gradient_norm < 1e-6 for d in diagnostics)
    assert all(
        d.initial_gradient_norm >= 1e-6 and d.gradient_norm < d.initial_gradient_norm
        for d in refined
    )
