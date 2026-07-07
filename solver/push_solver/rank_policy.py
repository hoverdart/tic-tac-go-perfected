"""Dependency-free push successor ranking policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_POLICY_PATH = Path(__file__).with_name("linear_push_ranker_v1.json")


@dataclass(frozen=True)
class LinearPushRankPolicy:
    name: str
    weights: Mapping[str, float]
    value_weights: Mapping[str, float]
    intercept: float = 0.0
    value_intercept: float = 0.0
    value_weight: float = 0.0

    def raw_score(self, features: Mapping[str, float]) -> float:
        return self.intercept + sum(
            self.weights.get(name, 0.0) * value
            for name, value in features.items()
        )

    def value(self, features: Mapping[str, float]) -> float:
        estimate = self.value_intercept + sum(
            self.value_weights.get(name, 0.0) * value
            for name, value in features.items()
        )
        return max(0.0, estimate)

    def score(self, features: Mapping[str, float]) -> float:
        return self.raw_score(features) - (self.value_weight * self.value(features))


_DEFAULT_POLICY: LinearPushRankPolicy | None = None


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> LinearPushRankPolicy:
    policy_path = Path(path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    weights = {
        str(name): float(value)
        for name, value in payload.get("weights", {}).items()
    }
    value_weights = {
        str(name): float(value)
        for name, value in payload.get("value_weights", {}).items()
    }
    return LinearPushRankPolicy(
        name=str(payload.get("name", policy_path.stem)),
        weights=weights,
        value_weights=value_weights,
        intercept=float(payload.get("intercept", 0.0)),
        value_intercept=float(payload.get("value_intercept", 0.0)),
        value_weight=float(payload.get("value_weight", 0.0)),
    )


def default_policy() -> LinearPushRankPolicy | None:
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is not None:
        return _DEFAULT_POLICY
    if not DEFAULT_POLICY_PATH.exists():
        return None
    _DEFAULT_POLICY = load_policy(DEFAULT_POLICY_PATH)
    return _DEFAULT_POLICY
