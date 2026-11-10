import numpy as np

from lease_lab.data.events import Event
from lease_lab.artifacts.store import FrozenBundle
from lease_lab.config import HISTORICAL_OFFERS
from lease_lab.contracts import Finance
from lease_lab.decisions.support import support_for
from lease_lab.report_contracts import CalibrationBin, PredictionScore, SliceScore


def losses(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("one finite prediction per binary factual response required")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("invalid predictions")
    # Numerical log-loss convention only, not confidence/variance manipulation.
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean((p - y) ** 2)), float(
        -np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe))
    )


def prediction_score(
    rows: list[Event],
    p: np.ndarray,
    family: str,
    model: str,
    cohort: str,
    uncertainty_method: str,
    *,
    bundle: FrozenBundle,
) -> PredictionScore:
    p = np.asarray(p, dtype=float)
    y = np.asarray([e.accepted for e in rows], dtype=float)
    brier, log_loss = losses(y, p)
    supported = sum(
        support_for(
            bundle.support,
            bundle.beta,
            e.context,
            Finance(),
            e.offer_id,
            e.card_id,
            e.card_id == "nov-2026" and e.offer_id in HISTORICAL_OFFERS,
        ).automated
        for e in rows
    )
    bins = []
    for i in range(8):
        low, high = i / 8, (i + 1) / 8
        mask = (p >= low) & (p < high if i < 7 else p <= high)
        bins.append(
            CalibrationBin(
                lower=low,
                upper=high,
                count=int(mask.sum()),
                prediction=float(p[mask].mean()) if mask.any() else None,
                observed=float(y[mask].mean()) if mask.any() else None,
            )
        )
    ece = sum(b.count * abs(b.prediction - b.observed) for b in bins if b.count) / len(
        rows
    )
    slices = []
    groups = {"segment:" + e.context.segment for e in rows} | {
        "offer:" + e.offer_id for e in rows
    }
    for group in sorted(groups):
        mask = np.array(
            [
                group in ("segment:" + e.context.segment, "offer:" + e.offer_id)
                for e in rows
            ]
        )
        b, l = losses(y[mask], p[mask])
        slices.append(SliceScore(slice=group, n=int(mask.sum()), brier=b, log_loss=l))
    return PredictionScore(
        family=family,
        model=model,
        cohort=cohort,
        n=len(rows),
        cohort_n=len(rows),
        advisory_supported_n=supported,
        advisory_unsupported_n=len(rows) - supported,
        missing_prediction_n=0,
        support_definition="advisory-support-heuristic-v1",
        brier=brier,
        log_loss=log_loss,
        calibration_error=ece,
        calibration=bins,
        slices=slices,
        uncertainty_method=uncertainty_method,
    )


def wilson(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(center - radius), float(center + radius)


def paired_bootstrap(
    values: np.ndarray, seed: int = 404, repeats: int = 500
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (repeats, len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))
