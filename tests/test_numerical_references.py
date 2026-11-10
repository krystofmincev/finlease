import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm

from lease_lab.evaluation.references import (
    ORDINARY_MAX_QUANTILE_ERROR,
    QUANTILE_LEVELS,
    compare_quantiles,
    integrate_reference,
    reference_fixtures,
)
from lease_lab.models import logistic_laplace


def independent_intercept_reference(successes, failures, sd):
    """Integrate after p=sigmoid(beta), including the change-of-variable Jacobian."""

    def density(p):
        if p <= 0.0 or p >= 1.0:
            return 0.0
        log_density = (
            successes * math.log(p)
            + failures * math.log1p(-p)
            - 0.5 * (logit(p) / sd) ** 2
            - math.log(sd * math.sqrt(2.0 * math.pi))
            - math.log(p)
            - math.log1p(-p)
        )
        return math.exp(log_density)

    evidence = quad(density, 0, 1, epsabs=1e-13, epsrel=1e-12)[0]
    quantiles = np.array(
        [
            brentq(
                lambda p: quad(density, 0, p, epsabs=1e-13, epsrel=1e-12)[0] / evidence
                - level,
                1e-12,
                1 - 1e-12,
            )
            for level in QUANTILE_LEVELS
        ]
    )
    return evidence, quantiles


def test_prior_predictive_quantiles_match_normal_projection():
    # No observations leave a known Gaussian prior and a normal linear projection.
    probes = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, -0.37], [0.0, 0.0]])
    prior_sd = np.array([1.0, 1.7])
    result = integrate_reference(np.empty((0, 2)), [], prior_sd, probes)
    projection_sd = np.sqrt(np.sum((probes * prior_sd) ** 2, axis=1))
    expected = expit(projection_sd[:, None] * norm.ppf(QUANTILE_LEVELS)[None, :])
    np.testing.assert_allclose(
        result.probability_quantiles, expected, rtol=0, atol=0.001
    )
    np.testing.assert_allclose(result.coefficient_mean, np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(
        result.coefficient_covariance, np.diag(prior_sd**2), atol=0.001
    )
    assert (
        result.diagnostics.log_evidence_lower_bound
        <= 0
        <= result.diagnostics.log_evidence_upper_bound
    )
    assert result.diagnostics.resolution_satisfactory


def test_one_dimensional_posterior_matches_transformed_adaptive_quadrature():
    successes, failures, sd = 7, 3, 1.4
    evidence, quantiles = independent_intercept_reference(successes, failures, sd)
    result = integrate_reference(
        np.ones((successes + failures, 1)),
        np.array([1] * successes + [0] * failures),
        sd,
        [1.0],
    )
    np.testing.assert_allclose(
        result.probability_quantiles[0], quantiles, rtol=0, atol=2e-5
    )
    assert math.exp(result.diagnostics.log_evidence_estimate) == pytest.approx(
        evidence, rel=2e-6
    )
    assert (
        result.diagnostics.log_evidence_lower_bound
        <= math.log(evidence)
        <= result.diagnostics.log_evidence_upper_bound
    )
    assert result.diagnostics.posterior_tail_probability_upper_bound < 1e-6
    assert result.diagnostics.resolution_satisfactory


def test_two_dimensions_match_independent_one_dimensional_quadratures():
    # The diagonal design separates the two coefficients and their likelihoods.
    X = np.array([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 12)
    y = np.array([1] * 7 + [0] * 3 + [1] * 4 + [0] * 8)
    sd = np.array([1.4, 1.1])
    evidence0, quantiles0 = independent_intercept_reference(7, 3, sd[0])
    evidence1, quantiles1 = independent_intercept_reference(4, 8, sd[1])
    result = integrate_reference(X, y, sd, np.eye(2))
    np.testing.assert_allclose(
        result.probability_quantiles,
        np.vstack([quantiles0, quantiles1]),
        rtol=0,
        atol=0.001,
    )
    assert math.exp(result.diagnostics.log_evidence_estimate) == pytest.approx(
        evidence0 * evidence1, rel=0.001
    )
    assert abs(result.coefficient_covariance[0, 1]) < 1e-10
    assert result.diagnostics.resolution_satisfactory


def test_correlated_projection_matches_nested_adaptive_quadrature():
    # Independently integrate over slope and projected logit, with unit Jacobian.
    sd, projection_slope = 1.25, 0.7
    X = np.column_stack((np.ones(12), np.repeat([-1.0, 0.0, 1.0], 4)))
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    result = integrate_reference(X, y, sd, [1.0, projection_slope])
    groups = [(-1.0, 1, 3), (0.0, 2, 2), (1.0, 3, 1)]

    def projected_density(projected_logit):
        def integrand(slope):
            intercept = projected_logit - projection_slope * slope
            exponent = (intercept**2 + slope**2) / (2 * sd**2)
            for feature, successes, failures in groups:
                eta = intercept + feature * slope
                exponent += successes * math.log1p(
                    math.exp(-eta)
                ) + failures * math.log1p(math.exp(eta))
            return math.exp(-exponent) / (2 * math.pi * sd**2)

        return quad(integrand, -11 * sd, 11 * sd, epsabs=1e-10, epsrel=1e-9)[0]

    limit = 11 * sd * (1 + projection_slope)
    evidence = quad(projected_density, -limit, limit, epsabs=1e-11, epsrel=1e-9)[0]
    expected = [
        expit(
            brentq(
                lambda z: quad(projected_density, -limit, z, epsabs=1e-11, epsrel=1e-8)[
                    0
                ]
                / evidence
                - level,
                -limit,
                limit,
                xtol=1e-8,
            )
        )
        for level in QUANTILE_LEVELS
    ]
    np.testing.assert_allclose(
        result.probability_quantiles[0], expected, rtol=0, atol=0.0003
    )
    assert (
        result.diagnostics.log_evidence_lower_bound
        <= math.log(evidence)
        <= result.diagnostics.log_evidence_upper_bound
    )
    assert result.diagnostics.resolution_satisfactory


def test_declared_fixture_references_converge_and_bound_truncation():
    fixtures = reference_fixtures()
    assert [fixture.category for fixture in fixtures] == [
        "ordinary",
        "ordinary",
        "sparse",
        "imbalanced",
        "near_separated",
        "separated",
    ]
    assert ORDINARY_MAX_QUANTILE_ERROR == 0.03
    for fixture in fixtures:
        assert set(fixture.y) == {0, 1}
        reference = integrate_reference(
            fixture.X, fixture.y, fixture.prior_sd, fixture.probes
        )
        assert reference.diagnostics.resolution_satisfactory, fixture.name
        assert reference.diagnostics.posterior_tail_probability_upper_bound <= 1e-6
        assert np.all(np.diff(reference.probability_quantiles, axis=1) >= 0)


def test_unreliable_reference_and_approximation_error_are_distinct():
    X, y, sd, probe = np.ones((10, 1)), [1] * 7 + [0] * 3, 1.4, [1.0]
    truncated = integrate_reference(X, y, sd, probe, bound_sd=0.5)
    assert truncated.diagnostics.posterior_tail_probability_upper_bound > 1e-6
    assert (
        compare_quantiles(truncated, truncated.probability_quantiles).status
        == "reference_unreliable"
    )
    reference = integrate_reference(X, y, sd, probe)
    assert (
        compare_quantiles(reference, reference.probability_quantiles).status == "pass"
    )
    assert (
        compare_quantiles(reference, [[0.1, 0.2, 0.3]]).status == "approximation_error"
    )


@pytest.mark.parametrize(
    "X, y, sd, probes",
    [
        (np.ones((3, 3)), [0, 1, 0], 1.0, [1, 1, 1]),
        (np.ones((3, 1)), [0, 1], 1.0, [1]),
        (np.ones((2, 1)), [0, np.nan], 1.0, [1]),
        (np.ones((2, 1)), [0, 1], 0.0, [1]),
        (np.ones((2, 1)), [0, 1], 1.0, [np.inf]),
    ],
)
def test_invalid_reference_inputs_fail(X, y, sd, probes):
    with pytest.raises(ValueError):
        integrate_reference(X, y, sd, probes)


def test_invalid_comparison_inputs_fail():
    reference = integrate_reference(np.ones((2, 1)), [0, 1], 1, [1])
    with pytest.raises(ValueError, match="shape"):
        compare_quantiles(reference, [0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="nondecreasing"):
        compare_quantiles(reference, [[0.8, 0.5, 0.9]])
    with pytest.raises(ValueError, match="finite probabilities"):
        compare_quantiles(reference, [[0, np.nan, 1]])


@pytest.mark.parametrize(
    "coefficients", [np.array([0.7, -1.1]), np.array([350.0, -230.0])]
)
def test_production_logistic_derivatives_agree_with_finite_differences(coefficients):
    logistic = logistic_laplace
    X = np.column_stack((np.ones(6), [-2.0, -1.0, 0.3, 1.5, 2.5, 4.0]))
    y = np.array([0, 1, 0, 1, 0, 1])
    prior_sd = 2.0
    step = 1e-4
    basis = np.eye(2)
    numerical_gradient = np.array(
        [
            (
                logistic.objective(coefficients + step * axis, X, y, prior_sd)
                - logistic.objective(coefficients - step * axis, X, y, prior_sd)
            )
            / (2 * step)
            for axis in basis
        ]
    )
    numerical_hessian = np.column_stack(
        [
            (
                logistic.gradient(coefficients + step * axis, X, y, prior_sd)
                - logistic.gradient(coefficients - step * axis, X, y, prior_sd)
            )
            / (2 * step)
            for axis in basis
        ]
    )
    assert np.isfinite(logistic.objective(coefficients, X, y, prior_sd))
    np.testing.assert_allclose(
        logistic.gradient(coefficients, X, y, prior_sd),
        numerical_gradient,
        rtol=1e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        logistic.hessian(coefficients, X, y, prior_sd),
        numerical_hessian,
        rtol=1e-6,
        atol=2e-6,
    )


def test_production_sampler_matches_inverse_hessian_orientation():
    logistic = logistic_laplace
    X = np.column_stack((np.ones(30), np.repeat([1.5, 2.0, 3.0], 10)))
    y = np.array([1] * 3 + [0] * 7 + [1] * 5 + [0] * 5 + [1] * 7 + [0] * 3)
    fit = logistic.fit_map(X, y, prior_sd=3.0)
    hessian = logistic.hessian(fit.mode, X, y, 3.0)
    covariance = np.linalg.inv(hessian)
    np.testing.assert_allclose(
        fit.precision_cholesky @ fit.precision_cholesky.T, hessian, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-9, atol=1e-9)
    # A correlated design exposes a reversed Cholesky solve, unlike a diagonal one.
    assert abs(covariance[0, 1]) / np.sqrt(covariance[0, 0] * covariance[1, 1]) > 0.8
    draws = logistic.sample_coefficients(fit, count=120_000, seed=37)
    assert draws.shape == (120_000, 2)
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), (draws - fit.mode).T).T
    np.testing.assert_allclose(whitened.mean(axis=0), np.zeros(2), atol=0.015)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(2), atol=0.02)


def test_production_predictive_quantiles_against_all_declared_fixtures():
    logistic = logistic_laplace
    for fixture in reference_fixtures():
        reference = integrate_reference(
            fixture.X, fixture.y, fixture.prior_sd, fixture.probes
        )
        # The production API has one common prior SD. Scaling design columns makes
        # that exactly equivalent to this fixture's independently specified SDs.
        common_sd = 3.0
        scale = fixture.prior_sd / common_sd
        fit = logistic.fit_map(fixture.X * scale, fixture.y, prior_sd=common_sd)
        probes = fixture.probes * scale
        projected_mean = probes @ fit.mode
        projected_variance = np.einsum("ij,jk,ik->i", probes, fit.covariance, probes)
        assert np.all(projected_variance > 0)
        approximate = expit(
            projected_mean[:, None]
            + np.sqrt(projected_variance)[:, None] * norm.ppf(QUANTILE_LEVELS)[None, :]
        )
        comparison = compare_quantiles(reference, approximate)
        assert comparison.reference_reliable, fixture.name
        if fixture.category == "ordinary":
            assert comparison.status == "pass", (fixture.name, comparison)
        if fixture.category in {"imbalanced", "near_separated", "separated"}:
            assert comparison.status == "approximation_error", (
                fixture.name,
                comparison,
            )
