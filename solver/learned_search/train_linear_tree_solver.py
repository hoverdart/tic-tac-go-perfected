"""Train the V1 linear tree ranker from optimized-solver expert paths."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from solver import optimized_solver
from solver.gymnasium_register import ranked_real_boards
from solver.learned_search.features import candidate_features
from solver.learned_search.linear_ranker import (
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    LinearRanker,
)
from solver.learned_search.training_data import expert_rows_for_solution


BOARD_GROUPS = {
    "five": ranked_real_boards.fiveBoards,
    "six": ranked_real_boards.sixBoards,
    "seven": ranked_real_boards.sevenBoards,
    "eight": ranked_real_boards.eightBoards,
    "nine": ranked_real_boards.nineBoards,
}


def _feature_vector(row: dict, means: dict[str, float], scales: dict[str, float]) -> list[float]:
    features = row["features"]
    return [
        (float(features[name]) - means[name]) / scales[name]
        for name in FEATURE_NAMES
    ]


def _dot(weights: list[float], vector: list[float], intercept: float = 0.0) -> float:
    return intercept + sum(weight * value for weight, value in zip(weights, vector))


def _load_rows(
    groups: list[str],
    limit_per_group: int | None,
    max_states: int,
    mode: str,
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    stats = {"solved": 0, "skipped": 0, "rows": 0}
    for group in groups:
        boards = BOARD_GROUPS[group]
        if limit_per_group is not None:
            boards = boards[:limit_per_group]
        for index, board in enumerate(boards):
            board_id = f"{group}-{index}"
            moves, _final_board, _states = optimized_solver.solve(
                board,
                progress_every=0,
                max_states=max_states,
                mode=mode,
            )
            if moves is None:
                stats["skipped"] += 1
                continue
            board_rows = expert_rows_for_solution(board, moves, board_id=board_id)
            rows.extend(board_rows)
            stats["solved"] += 1
            stats["rows"] += len(board_rows)
    return rows, stats


def _group_rows(rows: Iterable[dict]) -> list[list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["board_id"], int(row["depth"]))].append(row)
    return [group for group in grouped.values() if any(row["label"] == 1 for row in group)]


def _feature_stats(rows: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = [float(row["features"][name]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[name] = mean
        scales[name] = math.sqrt(variance) or 1.0
    return means, scales


def _train_pairwise_ranker(
    groups: list[list[dict]],
    means: dict[str, float],
    scales: dict[str, float],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[list[float], float]:
    random.seed(seed)
    weights = [0.0 for _ in FEATURE_NAMES]
    intercept = 0.0

    for _epoch in range(epochs):
        random.shuffle(groups)
        for group in groups:
            positives = [row for row in group if row["label"] == 1]
            negatives = [row for row in group if row["label"] == 0]
            if not positives or not negatives:
                continue
            positive_vector = _feature_vector(positives[0], means, scales)
            for negative in negatives:
                negative_vector = _feature_vector(negative, means, scales)
                diff = [
                    positive_value - negative_value
                    for positive_value, negative_value in zip(positive_vector, negative_vector)
                ]
                margin = _dot(weights, diff)
                probability = 1.0 / (
                    1.0 + math.exp(-max(-60.0, min(60.0, margin)))
                )
                gradient_scale = 1.0 - probability
                for index, value in enumerate(diff):
                    weights[index] = (
                        (1.0 - learning_rate * l2) * weights[index]
                        + learning_rate * gradient_scale * value
                    )

    return weights, intercept


def _ranking_accuracy(
    groups: list[list[dict]],
    weights: list[float],
    intercept: float,
    means: dict[str, float],
    scales: dict[str, float],
) -> dict[str, float]:
    correct = 0
    reciprocal_ranks = []
    for group in groups:
        scored = sorted(
            (
                (
                    _dot(weights, _feature_vector(row, means, scales), intercept),
                    int(row["label"]),
                )
                for row in group
            ),
            reverse=True,
        )
        if scored and scored[0][1] == 1:
            correct += 1
        for rank, (_score, label) in enumerate(scored, start=1):
            if label == 1:
                reciprocal_ranks.append(1.0 / rank)
                break
    total = len(groups)
    return {
        "top1_accuracy": correct / total if total else 0.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
        "groups": float(total),
    }


def train(
    *,
    output_path: Path = DEFAULT_MODEL_PATH,
    groups: list[str],
    limit_per_group: int | None,
    max_states: int,
    mode: str,
    holdout_fraction: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> dict[str, object]:
    rows, export_stats = _load_rows(groups, limit_per_group, max_states, mode)
    if not rows:
        raise RuntimeError("No training rows were generated.")

    all_groups = _group_rows(rows)
    random.seed(seed)
    random.shuffle(all_groups)
    split_index = max(1, int(len(all_groups) * (1.0 - holdout_fraction)))
    train_groups = all_groups[:split_index]
    holdout_groups = all_groups[split_index:] or all_groups[:]
    train_rows = [row for group in train_groups for row in group]
    means, scales = _feature_stats(train_rows)
    weights, intercept = _train_pairwise_ranker(
        train_groups,
        means,
        scales,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )

    metrics = {
        "train": _ranking_accuracy(train_groups, weights, intercept, means, scales),
        "holdout": _ranking_accuracy(holdout_groups, weights, intercept, means, scales),
    }
    ranker = LinearRanker(
        weights=dict(zip(FEATURE_NAMES, weights)),
        intercept=intercept,
        means=means,
        scales=scales,
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trainer": "pairwise_logistic_sgd",
            "groups": groups,
            "limit_per_group": limit_per_group,
            "max_states": max_states,
            "expert_mode": mode,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
            "export_stats": export_stats,
            "metrics": metrics,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ranker.to_json_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"output_path": str(output_path), "export_stats": export_stats, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Linear Tree Solver V1 ranker.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(BOARD_GROUPS),
        default=["five", "six"],
    )
    parser.add_argument("--limit-per-group", type=int, default=None)
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument("--mode", choices=("hybrid", "fast", "exact"), default="hybrid")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = train(
        output_path=args.output,
        groups=args.groups,
        limit_per_group=args.limit_per_group,
        max_states=args.max_states,
        mode=args.mode,
        holdout_fraction=args.holdout_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
