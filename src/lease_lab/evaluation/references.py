"""Independent, low-dimensional references for Gaussian-prior logistic models.

These routines are deliberately independent of the fitted model implementation.
They integrate one or two *independent zero-mean* Gaussian coefficients using
Gaussian-prior cell masses and midpoint likelihoods.  The result is a reference
for selected predictive probability quantiles, not validation of higher-dimensional
Laplace posteriors or of the data-generating process.

The default checks below were fixed before comparing the declared fixtures with
the Laplace implementation.  A refined-grid difference is an empirical diagnostic,
not a guaranteed quadrature-error bound.  The log-evidence interval and tail-mass
bound use a conservative Lipschitz argument, up to floating-point arithmetic.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit, log_ndtr, logsumexp, ndtr


FloatArray = NDArray[np.float64]
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
ORDINARY_MAX_QUANTILE_ERROR = 0.03
# These are reference-resolution diagnostics, set tighter than the model tolerance.
MAX_GRID_QUANTILE_DIFFERENCE = 0.003
MAX_GRID_LOG_EVIDENCE_DIFFERENCE = 0.001
MAX_POSTERIOR_TAIL_BOUND = 1e-6


@dataclass(frozen=True, slots=True)
class ReferenceFixture:
    name: str
    category: Literal["ordinary", "sparse", "imbalanced", "near_separated", "separated"]
    X: FloatArray
    y: FloatArray
    prior_sd: FloatArray
    probes: FloatArray


@dataclass(frozen=True, slots=True)
class ReferenceDiagnostics:
    fine_grid_points_per_axis: int
    coarse_grid_points_per_axis: int
    bounds: tuple[tuple[float, float], ...]
    log_evidence_estimate: float
    log_evidence_lower_bound: float
    log_evidence_upper_bound: float
    posterior_tail_probability_upper_bound: float
    log_evidence_grid_difference: float
    quantile_grid_differences: FloatArray
    max_quantile_grid_difference: float
    cell_log_likelihood_variation_bound: float
    resolution_satisfactory: bool


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    # Shape [probe, quantile], with quantiles ordered as QUANTILE_LEVELS.
    probability_quantiles: FloatArray
    coefficient_mean: FloatArray
    coefficient_covariance: FloatArray
    diagnostics: ReferenceDiagnostics


@dataclass(frozen=True, slots=True)
class QuantileComparison:
    max_absolute_error: float
    tolerance: float
    reference_reliable: bool
    passed: bool
    status: Literal["pass", "reference_unreliable", "approximation_error"]


def reference_fixtures() -> tuple[ReferenceFixture, ...]:
    """Return the fixed ordinary and stress fixtures, including both outcomes.

    Every row is a factual Bernoulli observation.  The one- and two-coefficient
    matrices are already the model design matrices; the first column is an
    intercept.  All priors are independent, including the proper intercept prior.
    Stress fixtures are allowed to pass or fail the tolerance; a failure is
    reported, never discarded to make the ordinary suite look better.
    """
    ordinary_x = np.column_stack((np.ones(60), np.repeat([-1.0, 0.0, 1.0], 20)))
    ordinary_y = np.array(
        [1.0] * 6 + [0.0] * 14 + [1.0] * 10 + [0.0] * 10 + [1.0] * 14 + [0.0] * 6
    )
    separated_x = np.column_stack((np.ones(20), np.repeat([-1.0, 1.0], 10)))
    return (
        ReferenceFixture(
            "ordinary_intercept",
            "ordinary",
            np.ones((40, 1)),
            np.array([1.0] * 24 + [0.0] * 16),
            np.array([1.5]),
            np.array([[1.0]]),
        ),
        ReferenceFixture(
            "ordinary_two_coefficients",
            "ordinary",
            ordinary_x,
            ordinary_y,
            np.array([1.5, 1.5]),
            np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]),
        ),
        ReferenceFixture(
            "sparse_intercept",
            "sparse",
            np.ones((3, 1)),
            np.array([1.0, 1.0, 0.0]),
            np.array([3.0]),
            np.array([[1.0]]),
        ),
        ReferenceFixture(
            "imbalanced_intercept",
            "imbalanced",
            np.ones((24, 1)),
            np.array([1.0] * 22 + [0.0] * 2),
            np.array([3.0]),
            np.array([[1.0]]),
        ),
        ReferenceFixture(
            "near_separated_two_coefficients",
            "near_separated",
            separated_x,
            np.array([0.0] * 9 + [1.0] + [1.0] * 10),
            np.array([2.0, 3.0]),
            np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]),
        ),
        ReferenceFixture(
            "separated_two_coefficients",
            "separated",
            separated_x.copy(),
            np.array([0.0] * 10 + [1.0] * 10),
            np.array([2.0, 3.0]),
            np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]),
        ),
    )


def _inputs(
    X: ArrayLike,
    y: ArrayLike,
    prior_sd: ArrayLike,
    probes: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)
    sd_array = np.asarray(prior_sd, dtype=float)
    probes_array = np.asarray(probes, dtype=float)
    if X_array.ndim != 2 or X_array.shape[1] not in (1, 2):
        raise ValueError("X must have one or two columns")
    dimensions = X_array.shape[1]
    if y_array.ndim != 1 or y_array.shape[0] != X_array.shape[0]:
        raise ValueError("y must be a vector with one outcome per row of X")
    if sd_array.ndim == 0:
        sd_array = np.full(dimensions, sd_array)
    if (
        sd_array.shape != (dimensions,)
        or not np.all(np.isfinite(sd_array))
        or np.any(sd_array <= 0)
    ):
        raise ValueError(
            "prior_sd must contain one finite positive standard deviation per coefficient"
        )
    if probes_array.ndim == 1:
        probes_array = probes_array.reshape(1, -1)
    if (
        probes_array.ndim != 2
        or probes_array.shape[0] == 0
        or probes_array.shape[1] != dimensions
    ):
        raise ValueError(
            "probes must have one or more rows with one value per coefficient"
        )
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(probes_array)):
        raise ValueError("X and probes must contain finite values")
    if not np.all(np.isin(y_array, (0.0, 1.0))):
        raise ValueError("y must contain only binary outcomes")
    return X_array, y_array, sd_array, probes_array


def _normal_cell_masses(edges: FloatArray) -> FloatArray:
    # The survival-function form avoids cancellation in the positive tail.
    lower, upper = edges[:-1], edges[1:]
    return np.where(lower >= 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))


def _weighted_quantiles(
    logits: FloatArray, weights: FloatArray, radii: FloatArray
) -> FloatArray:
    """Invert a piecewise-uniform cell approximation to the projected logit.

    Uniformly distributing each cell's mass through its rectangle avoids treating
    all of a cell's mass as a point. In two dimensions a projection can otherwise
    become a lattice, including when equal-sized coefficients are combined. The
    sum of two projected uniforms has a trapezoidal density. This interpolation
    improves resolution, but remains part of the numerical approximation.
    """
    a = float(np.max(radii))
    b = float(np.min(radii)) if len(radii) == 2 else 0.0
    if a == 0:
        return np.full(len(QUANTILE_LEVELS), 0.5)
    order = np.argsort(logits)
    sorted_logits = logits[order]
    sorted_weights = weights[order]
    cumulative = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    # Cumulative summation can drift slightly from one in floating point.
    weight_sum = cumulative[-1]
    cumulative /= weight_sum
    sorted_weights = sorted_weights / weight_sum

    def cdf(value: float) -> float:
        lower = np.searchsorted(sorted_logits, value - a - b, side="right")
        upper = np.searchsorted(sorted_logits, value + a + b, side="left")
        offset = value - sorted_logits[lower:upper]
        if b == 0:
            partial = (offset + a) / (2.0 * a)
        else:
            partial = np.where(
                offset < -a + b,
                (offset + a + b) ** 2 / (8.0 * a * b),
                np.where(
                    offset < a - b,
                    (offset + a) / (2.0 * a),
                    1.0 - (a + b - offset) ** 2 / (8.0 * a * b),
                ),
            )
        return float(cumulative[lower] + np.sum(sorted_weights[lower:upper] * partial))

    lower = float(sorted_logits[0] - a - b)
    upper = float(sorted_logits[-1] + a + b)
    return expit(
        np.array(
            [
                brentq(lambda value: cdf(value) - level, lower, upper)
                for level in QUANTILE_LEVELS
            ]
        )
    )


def _grid(
    X: FloatArray,
    y: FloatArray,
    sd: FloatArray,
    probes: FloatArray,
    bound_sd: float,
    points: int,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    dimensions = X.shape[1]
    edges = np.linspace(-bound_sd, bound_sd, points + 1)
    standardized_centers = (edges[:-1] + edges[1:]) / 2.0
    centers = [standardized_centers * s for s in sd]
    masses = _normal_cell_masses(edges)
    if not np.all(masses > 0):
        raise ValueError("integration cells are too small to resolve in floating point")
    log_masses = np.log(masses)
    if dimensions == 1:
        log_weights = log_masses.copy()
    else:
        log_weights = log_masses[:, None] + log_masses[None, :]
    for row, outcome in zip(X, y, strict=True):
        score = row[0] * centers[0]
        if dimensions == 2:
            score = score[:, None] + row[1] * centers[1][None, :]
        signed_score = (2.0 * outcome - 1.0) * score
        log_weights -= np.logaddexp(0.0, -signed_score)
    log_evidence = float(logsumexp(log_weights))
    if not np.isfinite(log_evidence):
        raise ValueError("log evidence is outside the finite numerical range")
    weights = np.exp(log_weights - log_evidence)

    quantiles = np.empty((len(probes), len(QUANTILE_LEVELS)))
    for i, probe in enumerate(probes):
        logits = probe[0] * centers[0]
        if dimensions == 2:
            logits = logits[:, None] + probe[1] * centers[1][None, :]
        radii = np.abs(probe) * bound_sd * sd / points
        quantiles[i] = _weighted_quantiles(logits.ravel(), weights.ravel(), radii)

    if dimensions == 1:
        mean = np.array([np.sum(weights * centers[0])])
        variance = np.sum(weights * (centers[0] - mean[0]) ** 2)
        covariance = np.array([[variance]])
    else:
        mean = np.array(
            [
                np.sum(weights.sum(axis=1) * centers[0]),
                np.sum(weights.sum(axis=0) * centers[1]),
            ]
        )
        centered_0, centered_1 = centers[0] - mean[0], centers[1] - mean[1]
        covariance = np.array(
            [
                [
                    np.sum(weights.sum(axis=1) * centered_0**2),
                    np.sum(weights * centered_0[:, None] * centered_1[None, :]),
                ],
                [
                    np.sum(weights * centered_0[:, None] * centered_1[None, :]),
                    np.sum(weights.sum(axis=0) * centered_1**2),
                ],
            ]
        )
    return quantiles, mean, covariance, log_evidence


def integrate_reference(
    X: ArrayLike,
    y: ArrayLike,
    prior_sd: ArrayLike,
    probes: ArrayLike,
    *,
    bound_sd: float = 11.0,
    grid_points: int | None = None,
) -> ReferenceResult:
    """Integrate a logistic posterior in one or two coefficients.

    ``prior_sd`` is scalar or one standard deviation per coefficient; the prior
    covariance is diagonal and its mean is zero. ``probes`` is a design vector or
    a matrix of design vectors. No-data inputs are useful for reference validation.
    The reported quantiles are of sigmoid(probe @ coefficient), not outcome noise.

    A separate half-sized grid gives empirical refinement differences.  For a grid
    cell with half-widths h_j, the log likelihood changes by at most
    R = sum_i sum_j |X_ij| h_j, because log(sigmoid(t)) is 1-Lipschitz. Thus the
    within-box normalizer lies between the midpoint estimate times exp(-R) and
    exp(R). Outside the box, likelihood <= 1 and Gaussian tail mass is bounded
    by a union bound. Their ratio bounds omitted posterior mass. These conservative
    bounds do not certify that the observed refinement difference bounds quantile
    error; inspect convergence and do not extrapolate beyond these low dimensions.
    """
    X_array, y_array, sd, probes_array = _inputs(X, y, prior_sd, probes)
    dimensions = X_array.shape[1]
    if not np.isfinite(bound_sd) or not 0 < bound_sd <= 12:
        raise ValueError("bound_sd must be finite and in (0, 12]")
    with np.errstate(over="ignore", invalid="ignore"):
        coefficient_extents = sd * bound_sd
        design_extents = np.sum(np.abs(X_array) * sd * bound_sd, axis=1)
        probe_extents = np.sum(np.abs(probes_array) * sd * bound_sd, axis=1)
    if not all(
        np.all(np.isfinite(value))
        for value in (coefficient_extents, design_extents, probe_extents)
    ):
        raise ValueError("integration range exceeds the finite numerical range")
    if grid_points is None:
        grid_points = 40001 if dimensions == 1 else 601
    maximum = 100001 if dimensions == 1 else 1001
    if (
        isinstance(grid_points, bool)
        or not isinstance(grid_points, int)
        or not 21 <= grid_points <= maximum
    ):
        raise ValueError(f"grid_points must be an integer between 21 and {maximum}")

    fine = _grid(X_array, y_array, sd, probes_array, bound_sd, grid_points)
    coarse_points = (grid_points + 1) // 2
    coarse = _grid(X_array, y_array, sd, probes_array, bound_sd, coarse_points)
    quantiles, mean, covariance, log_evidence = fine
    quantile_differences = np.abs(quantiles - coarse[0])
    max_quantile_difference = float(quantile_differences.max())
    evidence_difference = abs(log_evidence - coarse[3])

    half_widths = bound_sd * sd / grid_points
    variation_bound = float(np.sum(np.abs(X_array) * half_widths))
    if not np.isfinite(variation_bound):
        raise ValueError(
            "integration variation bound exceeds the finite numerical range"
        )
    log_lower = log_evidence - variation_bound
    # Bound the prior probability outside the box by a union of Gaussian tails.
    log_prior_tail_bound = min(0.0, float(np.log(2 * dimensions) + log_ndtr(-bound_sd)))
    log_upper = min(
        0.0, float(np.logaddexp(log_evidence + variation_bound, log_prior_tail_bound))
    )
    tail_bound = float(np.exp(min(0.0, log_prior_tail_bound - log_lower)))
    diagnostics = ReferenceDiagnostics(
        fine_grid_points_per_axis=grid_points,
        coarse_grid_points_per_axis=coarse_points,
        bounds=tuple((-float(bound_sd * s), float(bound_sd * s)) for s in sd),
        log_evidence_estimate=log_evidence,
        log_evidence_lower_bound=log_lower,
        log_evidence_upper_bound=log_upper,
        posterior_tail_probability_upper_bound=tail_bound,
        log_evidence_grid_difference=evidence_difference,
        quantile_grid_differences=quantile_differences,
        max_quantile_grid_difference=max_quantile_difference,
        cell_log_likelihood_variation_bound=variation_bound,
        resolution_satisfactory=(
            max_quantile_difference <= MAX_GRID_QUANTILE_DIFFERENCE
            and evidence_difference <= MAX_GRID_LOG_EVIDENCE_DIFFERENCE
            and tail_bound <= MAX_POSTERIOR_TAIL_BOUND
        ),
    )
    return ReferenceResult(quantiles, mean, covariance, diagnostics)


def compare_quantiles(
    reference: ReferenceResult,
    approximate_quantiles: ArrayLike,
    *,
    tolerance: float = ORDINARY_MAX_QUANTILE_ERROR,
) -> QuantileComparison:
    """Compare like-for-like predictive quantiles, refusing unreliable references."""
    approximate = np.asarray(approximate_quantiles, dtype=float)
    if approximate.shape != reference.probability_quantiles.shape:
        raise ValueError(
            "approximate quantiles must have the same probe and quantile shape as the reference"
        )
    if not np.all(np.isfinite(approximate)) or np.any(
        (approximate < 0) | (approximate > 1)
    ):
        raise ValueError("approximate quantiles must be finite probabilities")
    if np.any(np.diff(approximate, axis=1) < 0):
        raise ValueError("approximate quantiles must be nondecreasing")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    difference = float(np.max(np.abs(approximate - reference.probability_quantiles)))
    reliable = reference.diagnostics.resolution_satisfactory
    status = (
        "reference_unreliable"
        if not reliable
        else "approximation_error"
        if difference > tolerance
        else "pass"
    )
    return QuantileComparison(
        difference, float(tolerance), reliable, status == "pass", status
    )
