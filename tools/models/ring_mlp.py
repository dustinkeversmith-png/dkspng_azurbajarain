"""2-4-2 ring generator. Shape checks and scoring run in the compiled engine."""

from __future__ import annotations

from typing import Any


class RingMlp:
    """Validate and score a 22-parameter ring generator."""

    SHAPES = {"layer1": (2, 4, 4), "layer2": (4, 2, 2)}

    def engine(self):
        from tools import tensor_engine

        return tensor_engine

    def normalize(self, weights: Any) -> dict[str, Any]:
        return self.engine().normalize_ring_candidate(weights)

    def score(self, weights: dict[str, Any], seed: int = 42, samples: int = 100) -> dict[str, float]:
        return self.engine().score_ring(weights, seed, samples)
