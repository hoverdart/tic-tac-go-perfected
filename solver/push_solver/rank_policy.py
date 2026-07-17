"""Dependency-free push successor ranking policy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_POLICY_PATH = Path(__file__).with_name("linear_push_ranker_v1.json")
OPTIONAL_POLICY_PATHS = (
    Path(__file__).with_name("linear_push_ranker_hard_tail_v1.json"),
    Path(__file__).with_name("linear_push_ranker_recovery_v1.json"),
    Path(__file__).with_name("linear_push_ranker_backfill_v1.json"),
)


@dataclass(frozen=True)
class LinearPushRankPolicy:
    name: str
    weights: Mapping[str, float]
    value_weights: Mapping[str, float]
    state_action_hints: Mapping[str, float]
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

    def action_bonus(self, board, state, pushes: tuple) -> float:
        if not pushes:
            return 0.0
        return self.state_action_hints.get(state_action_key(board, state, pushes), 0.0)


@dataclass(frozen=True)
class EnsemblePushRankPolicy:
    policies: tuple[LinearPushRankPolicy, ...]

    @property
    def name(self) -> str:
        return "+".join(policy.name for policy in self.policies)

    def raw_score(self, features: Mapping[str, float]) -> float:
        return self.policies[0].raw_score(features)

    def value(self, features: Mapping[str, float]) -> float:
        return self.policies[0].value(features)

    def score(self, features: Mapping[str, float]) -> float:
        return self.policies[0].score(features)

    def action_bonus(self, board, state, pushes: tuple) -> float:
        return max(policy.action_bonus(board, state, pushes) for policy in self.policies)


_DEFAULT_POLICY: LinearPushRankPolicy | EnsemblePushRankPolicy | None = None
_DEFAULT_POLICY_PATHS: tuple[Path, ...] | None = None


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
    state_action_hints = {
        str(name): float(value)
        for name, value in payload.get("state_action_hints", {}).items()
    }
    return LinearPushRankPolicy(
        name=str(payload.get("name", policy_path.stem)),
        weights=weights,
        value_weights=value_weights,
        state_action_hints=state_action_hints,
        intercept=float(payload.get("intercept", 0.0)),
        value_intercept=float(payload.get("value_intercept", 0.0)),
        value_weight=float(payload.get("value_weight", 0.0)),
    )


def default_policy() -> LinearPushRankPolicy | EnsemblePushRankPolicy | None:
    global _DEFAULT_POLICY, _DEFAULT_POLICY_PATHS
    configured_path = os.getenv("PUSH_RANK_POLICY_PATH")
    primary_path = Path(configured_path) if configured_path else DEFAULT_POLICY_PATH
    policy_paths = (primary_path, *OPTIONAL_POLICY_PATHS)
    if _DEFAULT_POLICY is not None and _DEFAULT_POLICY_PATHS == policy_paths:
        return _DEFAULT_POLICY
    if not primary_path.exists():
        if configured_path:
            raise FileNotFoundError(
                f"PUSH_RANK_POLICY_PATH does not exist: {primary_path}"
            )
        return None
    policies = [load_policy(primary_path)]
    for path in OPTIONAL_POLICY_PATHS:
        if path.exists():
            policies.append(load_policy(path))
    if len(policies) == 1:
        _DEFAULT_POLICY = policies[0]
    else:
        _DEFAULT_POLICY = EnsemblePushRankPolicy(tuple(policies))
    _DEFAULT_POLICY_PATHS = policy_paths
    return _DEFAULT_POLICY


def _cells_key(cells) -> str:
    return ".".join(str(cell) for cell in sorted(cells))


def state_action_key(board, state, pushes: tuple) -> str:
    push = pushes[0]
    return "|".join(
        (
            f"{board.rows}x{board.cols}",
            _cells_key(board.walls),
            str(state.player),
            _cells_key(state.os),
            _cells_key(state.xs),
            f"{push.piece}:{push.cell}:{push.move}",
        )
    )
