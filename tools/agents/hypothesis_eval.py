"""Tournament tools for 2-4-2 ring hypotheses."""

from __future__ import annotations

import json
from typing import Any

from tools.core.config import MAX_CANDIDATES
from tools.models.ring_mlp import RingMlp


class HypothesisEval:
    """Stage, score, and reduce 2-4-2 ring candidates."""

    def __init__(self, model: RingMlp | None = None) -> None:
        self.model = model or RingMlp()

    def stage_hypotheses(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the candidate list for a for_each scorer. Does not run the network."""
        if not isinstance(args, dict):
            raise TypeError("stage_hypotheses expects an arguments object")
        return self.model.engine().stage_ring_candidates(args.get("candidates"), MAX_CANDIDATES)

    def evaluate_weight_hypothesis(self, args: dict[str, Any]) -> dict[str, Any]:
        """Measure ring loss for one candidate on 100 fixed N(0, 1) samples."""
        if not isinstance(args, dict):
            raise TypeError("evaluate_weight_hypothesis expects an arguments object")
        if args.get("target_mode", "ring") != "ring":
            raise ValueError("target_mode must be ring")
        weights = self.model.normalize(args.get("candidate_weights"))
        score = self.model.score(weights)
        return {
            "candidate_id": str(args.get("candidate_id", "candidate")),
            "loss": score["loss"],
            "mean_radius": score["mean_radius"],
            "min_radius": score["min_radius"],
            "max_radius": score["max_radius"],
            "std_radius": score["std_radius"],
            "parameter_count": 22,
            "weights": weights,
        }

    def select_best_hypothesis(self, args: dict[str, Any]) -> dict[str, Any]:
        """Reduce parallel ring scores to one winner."""
        if not isinstance(args, dict):
            raise TypeError("select_best_hypothesis expects an arguments object")
        raw = args.get("trials")
        if isinstance(raw, str):
            raw = json.loads(raw)
        return self.model.engine().select_best_hypothesis(raw)
