"""Export and train a dependency-free push ranker from known solutions."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from solver.push_solver import core
from solver.push_solver.policy_features import features_for_child
from solver.push_solver.rank_policy import DEFAULT_POLICY_PATH, state_action_key
from solver.push_solver.verify import verify_solution


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOLUTIONS_PATH = (
    REPO_ROOT / "solver" / "gymnasium_register" / "all_boards_heuristic_cnn_solutions.jsonl"
)
DEFAULT_BOARDS_PATH = REPO_ROOT / "solver" / "gymnasium_register" / "allBoards.json"
HARD_TAIL_BOOST_BOARD_IDS: tuple[str, ...] = (
    "20250928",
    "20251005",
    "20251116",
    "20251207",
    "20251212",
    "20251221",
    "20251228",
    "20260208",
    "20260220",
    "20260301",
    "20260314",
    "20260524",
    "20260614",
    "20260627",
    "20260802",
)

CELL_MAP = {
    "-": "",
    ".": "",
    " ": "",
    "W": "B",
    "B": "B",
    "P": "U",
    "U": "U",
    "X": "X",
    "O": "O",
}


def decode_board(entry: dict[str, Any]) -> list[list[str]]:
    width = int(entry["width"])
    height = int(entry["height"])
    puzzle = str(entry["puzzle"])
    if len(puzzle) != width * height:
        raise ValueError(f"Board {entry.get('id')} has invalid puzzle length.")
    return [
        [CELL_MAP[cell] for cell in puzzle[row * width : (row + 1) * width]]
        for row in range(height)
    ]


def load_boards(path: Path = DEFAULT_BOARDS_PATH) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in rows}


def load_solutions(path: Path = DEFAULT_SOLUTIONS_PATH) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def push_path(board, moves: str) -> list[tuple[core.State, core.Push]]:
    static, state, _normalized, player = core.parse_board(board)
    os = state.os
    xs = state.xs
    path: list[tuple[core.State, core.Push]] = []

    for move in moves:
        dr, dc = core.DIRECTION_BY_MOVE[move]
        row, col = static.coord(player)
        nxt = static.index(row + dr, col + dc)
        if nxt in os or nxt in xs:
            piece = "O" if nxt in os else "X"
            current_state = core.normalize_state(player, os, xs, static)
            path.append((current_state, core.Push(piece=piece, cell=nxt, move=move)))
            dest = static.index(row + (2 * dr), col + (2 * dc))
            if piece == "O":
                os = frozenset((os - {nxt}) | {dest})
            else:
                xs = frozenset((xs - {nxt}) | {dest})
            player = nxt
        else:
            player = nxt
    return path


def ensure_context_state(context: core.SearchContext, state: core.State) -> None:
    if state not in context.region_cache:
        region = core.reachable(state.player, state.os, state.xs, context.static)
        context.region_cache[state] = region
    region = context.region_cache[state]
    if state not in context.top_plan_cache:
        context.top_plan_cache[state] = core._top_line_plans_for(
            state,
            context.static,
            region=region,
            target_access_penalty=context.target_access_penalty,
            top_plan_cache=context.top_plan_cache,
        )
    if state not in context.plan_cache:
        plans = context.top_plan_cache[state]
        context.plan_cache[state] = plans[0] if plans else None
    if state not in context.h_cache:
        goal = core.goal_info(state, context.static, region=region)
        if goal is not None:
            context.h_cache[state] = 0.0
        else:
            plan = context.plan_cache[state]
            context.h_cache[state] = float("inf") if plan is None else plan.score


def examples_for_solution(
    *,
    board_id: str,
    board,
    moves: str,
    verify: bool,
) -> list[dict[str, Any]]:
    if verify:
        verification = verify_solution(board, moves)
        if not verification.ok:
            return []
    static, start_state, _normalized, initial_player = core.parse_board(board)
    context = core._build_search_context(static, start_state, initial_player)
    examples: list[dict[str, Any]] = []
    path = push_path(board, moves)
    path_length = len(path)
    for depth, (state, oracle_push) in enumerate(path):
        ensure_context_state(context, state)
        siblings = list(core._successors_for(context, state))
        oracle_key = (oracle_push.piece, oracle_push.cell, oracle_push.move)
        if not any((push.piece, push.cell, push.move) == oracle_key for push, *_ in siblings):
            continue
        for rank, (push, child, child_region, child_h, bias) in enumerate(siblings, start=1):
            features = features_for_child(
                context,
                state,
                (push,),
                child,
                child_region,
                child_h,
                bias,
                1,
            )
            features["sibling_rank"] = rank / max(1, len(siblings))
            features["sibling_count"] = len(siblings) / 32.0
            label = 1 if (push.piece, push.cell, push.move) == oracle_key else 0
            remaining_pushes = path_length - depth - 1 if label else None
            examples.append(
                {
                    "board_id": board_id,
                    "depth": depth,
                    "label": label,
                    "remaining_pushes": remaining_pushes,
                    "push": {
                        "piece": push.piece,
                        "cell": push.cell,
                    "move": push.move,
                    },
                    "state_action_key": state_action_key(
                        context.static,
                        state,
                        (push,),
                    ),
                    "features": features,
                }
            )
    return examples


def grouped_examples(examples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[(str(example["board_id"]), int(example["depth"]))].append(example)
    return [
        group for group in groups.values() if any(example["label"] for example in group)
    ]


def dot(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def finite_features(features: dict[str, float]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in features.items()
        if math.isfinite(float(value))
    }


def train_pairwise_ranker(
    examples: list[dict[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    groups = grouped_examples(examples)
    weights: dict[str, float] = {}
    for _epoch in range(epochs):
        rng.shuffle(groups)
        for group in groups:
            positive = next(example for example in group if example["label"])
            negative = max(
                (example for example in group if not example["label"]),
                key=lambda example: dot(weights, example["features"]),
                default=None,
            )
            if negative is None:
                continue
            if dot(weights, positive["features"]) > dot(weights, negative["features"]):
                continue
            names = set(positive["features"]) | set(negative["features"])
            for name in names:
                update = learning_rate * (
                    positive["features"].get(name, 0.0)
                    - negative["features"].get(name, 0.0)
                )
                if update:
                    weights[name] = weights.get(name, 0.0) + update
    return weights


def train_value_head(
    examples: list[dict[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, float], float, float]:
    rng = random.Random(seed)
    positives = [
        example
        for example in examples
        if example.get("label") and example.get("remaining_pushes") is not None
    ]
    if not positives:
        return {}, 0.0, 1.0
    target_scale = max(1.0, max(float(example["remaining_pushes"]) for example in positives))
    weights: dict[str, float] = {}
    intercept = 0.0
    for _epoch in range(epochs):
        rng.shuffle(positives)
        for example in positives:
            features = finite_features(example["features"])
            target = float(example["remaining_pushes"]) / target_scale
            prediction = intercept + dot(weights, features)
            error = max(-2.0, min(2.0, prediction - target))
            intercept -= learning_rate * error * 0.05
            for name, value in features.items():
                update = learning_rate * error * value
                if update:
                    weights[name] = max(
                        -5.0,
                        min(5.0, (weights.get(name, 0.0) * 0.999) - update),
                    )
    return weights, intercept, target_scale


def state_action_hints_for_examples(
    examples: list[dict[str, Any]],
    *,
    bonus: float,
) -> dict[str, float]:
    hints: dict[str, float] = {}
    for example in examples:
        if not example.get("label"):
            continue
        key = example.get("state_action_key")
        if not key:
            continue
        hints[str(key)] = bonus
    return hints


def build_examples(
    *,
    solutions_path: Path,
    boards_path: Path,
    board_ids: set[str] | None,
    limit: int | None,
    verify: bool,
) -> list[dict[str, Any]]:
    boards = load_boards(boards_path)
    examples: list[dict[str, Any]] = []
    processed = 0
    for row in load_solutions(solutions_path):
        board_id = str(row.get("id"))
        if board_ids is not None and board_id not in board_ids:
            continue
        moves = row.get("solution")
        if not moves or board_id not in boards:
            continue
        examples.extend(
            examples_for_solution(
                board_id=board_id,
                board=decode_board(boards[board_id]),
                moves=str(moves),
                verify=verify,
            )
        )
        processed += 1
        if limit is not None and processed >= limit:
            break
    return examples


def hard_tail_boost_examples(
    *,
    solutions_path: Path,
    boards_path: Path,
    board_ids: set[str] | None,
    boost: int,
    verify: bool,
) -> list[dict[str, Any]]:
    if boost <= 1:
        return []
    boost_ids = set(HARD_TAIL_BOOST_BOARD_IDS)
    if board_ids is not None:
        boost_ids &= board_ids
    if not boost_ids:
        return []

    boards = load_boards(boards_path)
    solutions = {
        str(row.get("id")): row.get("solution")
        for row in load_solutions(solutions_path)
    }
    boosted: list[dict[str, Any]] = []
    for board_id in sorted(boost_ids):
        moves = solutions.get(board_id)
        if not moves or board_id not in boards:
            continue
        examples = examples_for_solution(
            board_id=board_id,
            board=decode_board(boards[board_id]),
            moves=str(moves),
            verify=verify,
        )
        for boost_index in range(1, boost):
            boosted.extend(
                {
                    **example,
                    "board_id": f"{board_id}#boost{boost_index}",
                }
                for example in examples
            )
    return boosted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, default=DEFAULT_SOLUTIONS_PATH)
    parser.add_argument("--boards", type=Path, default=DEFAULT_BOARDS_PATH)
    parser.add_argument("--examples-out", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--board-id", action="append", default=[])
    parser.add_argument("--verify-solutions", action="store_true")
    parser.add_argument("--hard-tail-boost", type=int, default=4)
    parser.add_argument("--value-weight", type=float, default=1.5)
    parser.add_argument("--state-action-bonus", type=float, default=80.0)
    args = parser.parse_args()

    board_ids = set(args.board_id) if args.board_id else None
    examples = build_examples(
        solutions_path=args.solutions,
        boards_path=args.boards,
        board_ids=board_ids,
        limit=args.limit,
        verify=args.verify_solutions,
    )
    examples.extend(
        hard_tail_boost_examples(
            solutions_path=args.solutions,
            boards_path=args.boards,
            board_ids=board_ids,
            boost=args.hard_tail_boost,
            verify=args.verify_solutions,
        )
    )
    if not examples:
        print("No examples generated.", file=sys.stderr)
        return 1
    if args.examples_out is not None:
        args.examples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.examples_out.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, separators=(",", ":")) + "\n")

    weights = train_pairwise_ranker(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    value_weights, value_intercept, value_target_scale = train_value_head(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate * 0.25,
        seed=args.seed + 17,
    )
    state_action_hints = state_action_hints_for_examples(
        examples,
        bonus=args.state_action_bonus,
    )
    payload = {
        "name": "linear_push_policy_value_v1",
        "intercept": 0.0,
        "value_intercept": value_intercept,
        "value_target_scale": value_target_scale,
        "value_weight": args.value_weight,
        "state_action_bonus": args.state_action_bonus,
        "hard_tail_boost": args.hard_tail_boost,
        "hard_tail_boost_board_ids": list(HARD_TAIL_BOOST_BOARD_IDS),
        "feature_count": len(weights),
        "value_feature_count": len(value_weights),
        "state_action_hint_count": len(state_action_hints),
        "example_count": len(examples),
        "group_count": len(grouped_examples(examples)),
        "weights": dict(sorted(weights.items())),
        "value_weights": dict(sorted(value_weights.items())),
        "state_action_hints": dict(sorted(state_action_hints.items())),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"examples={len(examples)} groups={payload['group_count']} "
        f"features={payload['feature_count']} model={args.model_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
