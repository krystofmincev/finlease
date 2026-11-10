"""Frozen train-only transform. Debt burden is pre-existing, fixed across offers."""

import numpy as np
from pydantic import model_validator

from lease_lab.contracts import Context, Finite, StrictModel
from lease_lab.data.events import Event


def raw(context: Context, offer_id: str) -> list[float]:
    return [
        context.business_age_years,
        float(np.log(context.monthly_turnover_cents / 100)),
        context.existing_debt_burden,
        (int(offer_id[1:]) - 900) / 100,
    ]


class Transform(StrictModel):
    mean: list[Finite]
    scale: list[Finite]
    version: str = "age-logturnover-preexistingburden-rate-segment-v1"

    @model_validator(mode="after")
    def dimensions(self):
        if len(self.mean) != 4 or len(self.scale) != 4 or min(self.scale) <= 0:
            raise ValueError("four positive-scale feature columns required")
        if self.version != "age-logturnover-preexistingburden-rate-segment-v1":
            raise ValueError("unknown feature contract")
        return self

    def matrix(
        self, contexts: list[Context], offer_ids: list[str] | tuple[str, ...]
    ) -> np.ndarray:
        if len(contexts) != len(offer_ids):
            raise ValueError("context/offer row mismatch")
        values = (
            np.asarray([raw(c, o) for c, o in zip(contexts, offer_ids)])
            - np.asarray(self.mean)
        ) / np.asarray(self.scale)
        segments = np.array(
            [
                [float(c.segment == s) for s in ("growing", "established", "restart")]
                for c in contexts
            ]
        )
        return np.column_stack([np.ones(len(contexts)), values, segments])


def fit_transform(rows: list[Event]) -> Transform:
    if not rows or any(e.split not in ("fit", "replay") for e in rows):
        raise ValueError("transform may fit only fit/replay factual rows")
    values = np.asarray([raw(e.context, e.offer_id) for e in rows])
    scales = values.std(axis=0)
    if (scales <= 0).any():
        raise ValueError("constant feature; transform cannot standardise")
    return Transform(mean=values.mean(axis=0).tolist(), scale=scales.tolist())
