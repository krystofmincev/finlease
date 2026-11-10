import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

from lease_lab.artifacts.store import load_bundle
from lease_lab.contracts import Context, DecisionRequest
from lease_lab.contracts import Choice
from lease_lab.data.features import fit_transform, raw
from lease_lab.decisions.policy import evaluate
from lease_lab.evaluation.benchmarks import numerical_checks
from lease_lab.models.beta_cells import fit_beta
from lease_lab.models.logistic_laplace import (
    LogisticSampler,
    fit_logistic,
    sample_coefficients,
)
from lease_lab.models.online import OnlineLearner
from lease_lab.simulation.environment import generate


def test_serving_and_learning_transitive_imports_exclude_truth():
    seen = set()

    def visit(module):
        if module in seen:
            return
        seen.add(module)
        assert not module.startswith(
            (
                "lease_lab.simulation",
                "lease_lab.evaluation",
                "lease_lab.offline",
                "torch",
                "tabpfn",
            )
        ), module
        spec = importlib.util.find_spec(module)
        if not spec or not spec.origin or not spec.origin.endswith(".py"):
            return
        tree = ast.parse(Path(spec.origin).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("lease_lab"):
                    visit(name)

    for root in (
        "lease_lab.api.app",
        "lease_lab.models.beta_cells",
        "lease_lab.models.logistic_laplace",
        "lease_lab.models.online",
    ):
        visit(root)


def test_fitting_cannot_consume_sealed_or_diagnostic_outcomes():
    rows = generate(200, 35)
    fit = [e for e in rows if e.split == "fit"]
    test = [e for e in rows if e.split == "test"]
    original = fit_logistic(fit)
    poisoned = [
        e.model_copy(
            update={
                "accepted": not e.accepted,
                "context": Context(monthly_turnover_cents=100000000),
            }
        )
        for e in test
    ]
    for fit_fn in (fit_logistic, fit_beta, fit_transform):
        with pytest.raises(ValueError):
            fit_fn(fit + poisoned)
    assert fit_logistic(fit).mode == original.mode
    assert fit_transform(fit).mean == original.transform.mean
    assert raw(Context(), "r800")[2] == raw(Context(), "r1000")[2]


def test_shared_logistic_coefficients_and_known_failure_fallback():
    b = load_bundle()
    if not b.laplace:
        pytest.fail("build --through pooled required")
    state = b.laplace["default"]
    context = Context()
    ids = ("r800", "r900", "r1000")
    draws = LogisticSampler(state).sample(context, ids, 4096, 8)
    coefficients = sample_coefficients(state.gaussian(), 4096, 8)
    from scipy.special import expit

    np.testing.assert_allclose(
        draws.probabilities,
        expit(coefficients @ state.transform.matrix([context] * 3, ids).T),
    )
    bad = state.model_copy(
        update={
            "diagnostics": state.diagnostics.model_copy(
                update={"reference_check": "failed"}
            )
        }
    )
    fallback = evaluate(
        DecisionRequest(method="laplace"),
        b.model_copy(update={"laplace": {"default": bad}}),
    )
    assert fallback.method == "beta" and any(
        "fallback" in s for s in fallback.diagnostics
    )
    checks = numerical_checks()
    assert all(
        c.fallback is not None and c.fallback.status == "review"
        for c in checks
        if c.status != "pass"
    )
    assert all(c.fallback is None for c in checks if c.category == "ordinary")


@pytest.mark.parametrize(
    "failure", [{"converged": False}, {"reference_check": "failed"}]
)
def test_known_failed_posterior_cannot_predict_or_choose_greedily(failure):
    state = load_bundle().laplace["default"]
    bad = state.model_copy(
        update={"diagnostics": state.diagnostics.model_copy(update=failure)}
    )
    sampler = LogisticSampler(bad)
    with pytest.raises(ValueError, match="numerical/reference gate"):
        sampler.predict([Context()], ["r900"])
    initial = [
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
    learner = OnlineLearner(initial, "laplace")
    learner.state = bad
    learner.sampler = sampler
    with pytest.raises(ValueError, match="numerical/reference gate"):
        learner.choose(Context(), "greedy", 7)
    with pytest.raises(ValueError, match="numerical/reference gate"):
        learner.width()


def test_independent_evaluator_counts_attempted_support_violations(monkeypatch):
    from lease_lab.evaluation import benchmarks

    bundle = load_bundle()
    event = next(e for e in generate(600, 17) if e.context.segment == "restart")
    original = evaluate(DecisionRequest(context=event.context), bundle)
    # Bypass only the producer validator, to test the evaluator independently.
    injected = original.model_copy(
        update={
            "greedy": Choice(
                status="recommend", offer_id="r1000", reason="injected-invalid-action"
            )
        }
    )
    monkeypatch.setattr(benchmarks, "evaluate", lambda request, bundle: injected)
    scores = benchmarks.policy_scores([event], bundle, "smooth")
    assert all(
        s.support_violations == 1 and s.reviews == 1
        for s in scores
        if s.policy == "greedy"
    )


def test_point_fitting_module_cannot_import_generator_or_cohort_owner():
    from lease_lab.evaluation import point_models
    import inspect

    tree = ast.parse(inspect.getsource(point_models))
    imports = [
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    ]
    assert not any(
        n.startswith("lease_lab.simulation") or n == "lease_lab.evaluation.cohorts"
        for n in imports
    )
    assert list(inspect.signature(point_models.develop_points).parameters) == [
        "fit",
        "early_stop",
        "selection",
        "fit_subsets",
        "settings",
    ]
