from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lease_lab.contracts import Context


@dataclass(frozen=True)
class Draws:
    probabilities: np.ndarray
    offer_ids: tuple[str, ...]
    method: str
    target: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        values = self.probabilities
        if (
            values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != len(self.offer_ids)
        ):
            raise ValueError("draw matrix/order mismatch")
        if len(set(self.offer_ids)) != len(self.offer_ids):
            raise ValueError("duplicate offer columns would break joint identity")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("non-finite/invalid probability draws")


class Sampler(Protocol):
    def sample(
        self, context: Context, offer_ids: tuple[str, ...], count: int, seed: int
    ) -> Draws: ...
