from __future__ import annotations

from ..generator.base import Generator


class BasePopulation(Generator):
    def __init__(
        self,
        batch_size: int = 1,
        max_samples: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(batch_size=batch_size, max_samples=max_samples, seed=seed)

    def next(self):
        raise NotImplementedError("Not implemented.")

    def update_state(self):
        raise NotImplementedError("Not implemented.")
