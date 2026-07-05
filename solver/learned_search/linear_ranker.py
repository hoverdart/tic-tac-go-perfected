"""Runtime linear ranker for learned tree-search guidance."""

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

DEFAULT_MODEL_PATH = Path(__file__).with_name("linear_tree_ranker_v1.json")


@dataclass(frozen=True)
class LinearRanker:
    """Linear child-path scorer used by the learned tree solver.

    The model scores one legal compressed child path from the current tree node.
    Higher scores mean the child should be explored earlier. `means` and
    `scales` are optional so old hand-written weight dictionaries still work.
    """

    weights: Mapping[str, float]
    intercept: float = 0.0
    means: Mapping[str, float] | None = None
    scales: Mapping[str, float] | None = None
    metadata: Mapping[str, object] | None = None

    def score(self, features: CandidateFeatures) -> float:
        values = features.to_dict()
        score = self.intercept
        for name in FEATURE_NAMES:
            value = float(values[name])
            if self.means is not None:
                value -= float(self.means.get(name, 0.0))
            if self.scales is not None:
                scale = float(self.scales.get(name, 1.0))
                if scale:
                    value /= scale
            score += self.weights.get(name, 0.0) * value
        return score

    @classmethod
    def default(cls) -> "LinearRanker":
        return cls(DEFAULT_WEIGHTS)

    @classmethod
    def v1(cls) -> "LinearRanker":
        """Load the trained V1 artifact when present, else use the baseline."""
        if DEFAULT_MODEL_PATH.exists():
            return cls.from_json(DEFAULT_MODEL_PATH)
        return cls.default()

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=payload.get("weights", {}),
            intercept=float(payload.get("intercept", 0.0)),
            means=payload.get("means"),
            scales=payload.get("scales"),
            metadata=payload.get("metadata"),
        )

    def to_json_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "model_type": "linear_tree_ranker",
            "feature_names": list(FEATURE_NAMES),
            "intercept": self.intercept,
            "weights": dict(self.weights),
            "means": dict(self.means or {}),
            "scales": dict(self.scales or {}),
            "metadata": dict(self.metadata or {}),
        }
