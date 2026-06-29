"""Tiny runtime ranker interface for learned search guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from solver.learned_search.features import CandidateFeatures


FEATURE_NAMES = (
    "parent_heuristic",
    "child_heuristic",
    "heuristic_delta",
    "segment_length",
    "pushed_o",
    "pushed_x",
    "walk_only",
    "child_solved",
    "child_lost",
    "child_pruned",
    "useful_line_occupancy",
    "x_threat_lines",
)


DEFAULT_WEIGHTS = {
    "heuristic_delta": 1.0,
    "child_heuristic": -0.15,
    "segment_length": -0.05,
    "pushed_o": 0.4,
    "pushed_x": -0.15,
    "walk_only": -0.1,
    "child_solved": 10.0,
    "child_lost": -10.0,
    "child_pruned": -10.0,
    "useful_line_occupancy": 0.8,
    "x_threat_lines": -0.6,
}


@dataclass(frozen=True)
class LinearRanker:
    """Simple scoring model used until a trained model artifact exists."""

    weights: Mapping[str, float]
    intercept: float = 0.0

    def score(self, features: CandidateFeatures) -> float:
        values = features.to_dict()
        return self.intercept + sum(
            self.weights.get(name, 0.0) * float(values[name])
            for name in FEATURE_NAMES
        )

    @classmethod
    def default(cls) -> "LinearRanker":
        return cls(DEFAULT_WEIGHTS)

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=payload.get("weights", {}),
            intercept=float(payload.get("intercept", 0.0)),
        )
