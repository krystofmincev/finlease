"""Proper Gaussian-prior logistic MAP and checked Laplace draws, CPU only."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import model_validator
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import expit

from lease_lab.contracts import Context, Finite, StrictModel
from lease_lab.data.events import Event, mature_rows, require_binary_outcomes
from lease_lab.data.features import Transform, fit_transform
from lease_lab.models.sampling import Draws

HERMITE_NODES, HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(24)


def objective(
    beta: np.ndarray, X: np.ndarray, y: np.ndarray, prior_sd: float = 3.0
) -> float:
    score = X @ beta
    return float(
        np.sum(np.logaddexp(0, score) - y * score) + 0.5 * (beta @ beta) / prior_sd**2
    )


def gradient(
    beta: np.ndarray, X: np.ndarray, y: np.ndarray, prior_sd: float = 3.0
) -> np.ndarray:
    return X.T @ (expit(X @ beta) - y) + beta / prior_sd**2


def hessian(
    beta: np.ndarray, X: np.ndarray, y: np.ndarray, prior_sd: float = 3.0
) -> np.ndarray:
    p = expit(X @ beta)
    return X.T @ ((p * (1 - p))[:, None] * X) + np.eye(X.shape[1]) / prior_sd**2


class FitDiagnostics(StrictModel):
    converged: bool
    optimizer_success: bool
    message: str
    gradient_norm: Finite
    condition_number: Finite
    min_precision_eigenvalue: Finite
    iterations: int
    rows: int
    coefficients: int
    reference_check: Literal["not_checked", "passed", "failed"] = "not_checked"
    initial_gradient_norm: Finite | None = None
    newton_refinement_steps: Literal[0, 1] = 0


def posterior_is_usable(diagnostics: FitDiagnostics) -> bool:
    return diagnostics.converged and diagnostics.reference_check != "failed"


def require_usable_posterior(diagnostics: FitDiagnostics) -> None:
    if not posterior_is_usable(diagnostics):
        raise ValueError("approximate posterior failed numerical/reference gate")


@dataclass(frozen=True)
class GaussianFit:
    mode: np.ndarray
    precision_cholesky: np.ndarray
    diagnostics: FitDiagnostics

    @property
    def covariance(self) -> np.ndarray:
        inverse_factor = solve_triangular(
            self.precision_cholesky, np.eye(len(self.mode)), lower=True
        )
        return inverse_factor.T @ inverse_factor


def fit_map(X: np.ndarray, y: np.ndarray, prior_sd: float = 3.0) -> GaussianFit:
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (
        X.ndim != 2
        or y.shape != (len(X),)
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(prior_sd)
        or prior_sd <= 0
    ):
        raise ValueError("invalid design, binary outcomes or Gaussian prior")
    result = minimize(
        objective,
        np.zeros(X.shape[1]),
        args=(X, y, prior_sd),
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-8, "maxiter": 100},
    )
    mode = result.x.copy()
    precision = hessian(mode, X, y, prior_sd)
    initial_norm = float(np.linalg.norm(gradient(mode, X, y, prior_sd), ord=np.inf))
    refinements = 0
    # Observed delayed/fixed replay seed 0 stopped at residual 1.14e-6 on
    # an objective plateau. One near-root Newton correction solves the same
    # strictly convex objective; no prior, gate, Hessian or variance is clipped.
    if 1e-6 <= initial_norm < 1e-3 and np.linalg.cond(precision) < 1e10:
        candidate = mode - np.linalg.solve(precision, gradient(mode, X, y, prior_sd))
        old_objective = objective(mode, X, y, prior_sd)
        roundoff = 32 * np.finfo(float).eps * max(1.0, abs(old_objective))
        candidate_norm = float(
            np.linalg.norm(gradient(candidate, X, y, prior_sd), ord=np.inf)
        )
        if (
            np.isfinite(candidate).all()
            and candidate_norm < initial_norm
            and objective(candidate, X, y, prior_sd) <= old_objective + roundoff
        ):
            mode = candidate
            refinements = 1
            precision = hessian(mode, X, y, prior_sd)
    factor = np.linalg.cholesky(precision)  # no arbitrary jitter or variance clipping
    norm = float(np.linalg.norm(gradient(mode, X, y, prior_sd), ord=np.inf))
    condition = float(np.linalg.cond(precision))
    diagnostics = FitDiagnostics(
        converged=bool(norm < 1e-6 and condition < 1e10),
        optimizer_success=bool(result.success),
        message=str(result.message),
        gradient_norm=norm,
        condition_number=condition,
        min_precision_eigenvalue=float(np.linalg.eigvalsh(precision)[0]),
        iterations=int(result.nit),
        rows=len(X),
        coefficients=X.shape[1],
        initial_gradient_norm=initial_norm,
        newton_refinement_steps=refinements,
    )
    return GaussianFit(mode, factor, diagnostics)


def sample_coefficients(fit: GaussianFit, count: int, seed: int) -> np.ndarray:
    require_usable_posterior(fit.diagnostics)
    z = np.random.default_rng(seed).standard_normal((count, len(fit.mode)))
    deviations = solve_triangular(fit.precision_cholesky.T, z.T, lower=False).T
    return fit.mode + deviations


class LogisticState(StrictModel):
    transform: Transform
    mode: list[Finite]
    precision_cholesky: list[list[Finite]]
    prior_sd: Finite
    diagnostics: FitDiagnostics

    @model_validator(mode="after")
    def valid_precision(self):
        L = np.asarray(self.precision_cholesky)
        if (
            len(self.mode) != 8
            or L.shape != (8, 8)
            or not np.array_equal(L, np.tril(L))
            or (np.diag(L) <= 0).any()
            or self.prior_sd <= 0
        ):
            raise ValueError("invalid frozen Laplace precision/dimensions")
        return self

    def gaussian(self) -> GaussianFit:
        return GaussianFit(
            np.asarray(self.mode), np.asarray(self.precision_cholesky), self.diagnostics
        )


def fit_logistic(
    rows: list[Event], prior_sd: float = 3.0, transform: Transform | None = None
) -> LogisticState:
    rows = mature_rows(rows)
    if any(e.split not in ("fit", "replay") for e in rows):
        raise ValueError("cannot fit on diagnostic/selection/test rows")
    require_binary_outcomes(rows)
    transform = transform or fit_transform(rows)
    X = transform.matrix([e.context for e in rows], [e.offer_id for e in rows])
    fitted = fit_map(X, np.asarray([e.accepted for e in rows], dtype=float), prior_sd)
    return LogisticState(
        transform=transform,
        mode=fitted.mode.tolist(),
        precision_cholesky=fitted.precision_cholesky.tolist(),
        prior_sd=prior_sd,
        diagnostics=fitted.diagnostics,
    )


class LogisticSampler:
    def __init__(self, state: LogisticState):
        self.state = state
        self._covariance = state.gaussian().covariance

    def sample(
        self, context: Context, offer_ids: tuple[str, ...], count: int, seed: int
    ) -> Draws:
        X = self.state.transform.matrix([context] * len(offer_ids), offer_ids)
        coefficients = sample_coefficients(self.state.gaussian(), count, seed)
        return Draws(
            expit(coefficients @ X.T),
            offer_ids,
            "gaussian-prior-map-laplace",
            "acceptance probability conditional on the full visible inputs",
            (
                "Laplace is a local Gaussian approximation, not an exact posterior.",
                "Low-dimensional reference checks do not certify high-dimensional uncertainty.",
                f"Precision condition number: {self.state.diagnostics.condition_number:.1f}.",
            ),
        )

    def predict(
        self, contexts: list[Context], offer_ids: list[str] | tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        require_usable_posterior(self.state.diagnostics)
        X = self.state.transform.matrix(contexts, offer_ids)
        mean = X @ np.asarray(self.state.mode)
        variance = np.einsum("ij,jk,ik->i", X, self._covariance, X)
        if (variance < 0).any():
            raise ValueError("invalid negative predictive variance")
        predictions = expit(
            mean[:, None] + np.sqrt(2 * variance)[:, None] * HERMITE_NODES
        )
        return (
            predictions @ HERMITE_WEIGHTS / np.sqrt(np.pi),
            expit(mean - 1.6448536269514722 * np.sqrt(variance)),
            expit(mean + 1.6448536269514722 * np.sqrt(variance)),
        )
