"""Offline feature-only ranker using the existing independent exporter/trainer.

Transforms inherit their original board's split. The nine release targets are
explicit development data, never validation. No exact state-action hints are
exported, and production model files cannot be overwritten by this command.
"""

from __future__ import annotations
import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver.benchmark_v4 import CORPUS, TARGETS, corpus, read_rows
from solver.push_solver.training_export import (
    examples_for_solution,
    train_pairwise_ranker,
    grouped_examples,
    dot,
)
from solver.push_solver.verify import verify_solution


def identity(row):
    """Group duplicate boards and all rotations/reflections before splitting."""
    width, height = row["width"], row["height"]
    board = tuple(
        tuple(row["puzzle"][r * width : (r + 1) * width]) for r in range(height)
    )
    variants = []
    for _ in range(4):
        for version in (board, tuple(tuple(reversed(r)) for r in board)):
            variants.append(
                f"{len(version)}x{len(version[0])}:"
                + "".join("".join(r) for r in version)
            )
        board = tuple(tuple(r) for r in zip(*board[::-1]))
    return sha256(min(variants).encode()).hexdigest()


def splits(rows):
    groups = {r["id"]: identity(r) for r in rows}
    dev = {groups[k] for k in TARGETS if k in groups}
    return {
        k: (
            "development"
            if group in dev
            else "validation"
            if int(group[:8], 16) % 5 == 0
            else "train"
        )
        for k, group in groups.items()
    }


def ranking_accuracy(examples, weights):
    groups = grouped_examples(examples)
    correct = sum(
        bool(max(group, key=lambda row: dot(weights, row["features"]))["label"])
        for group in groups
    )
    return {
        "decisions": len(groups),
        "correct": correct,
        "accuracy": correct / len(groups) if groups else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--paths", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--max-boards", type=int)
    a = p.parse_args()
    if a.output.resolve().parent == Path(__file__).resolve().parent:
        p.error("Candidate must stay outside the production model directory")
    rows = corpus(a.corpus)
    boards = {r["id"]: r for r in rows}
    split = splits(rows)
    paths = {}
    for path in a.paths:
        for row in read_rows(path):
            board_id = str(
                row.get("original_id")
                or row.get("id")
                or row.get("date", "").replace("-", "")
            )
            entry = boards.get(board_id)
            if entry is None:
                continue
            # Transformed artifacts must provide a source identity and the
            # original path. Only original coordinates are exported here.
            if row.get("transformation"):
                continue
            if row.get("board_sha256") != entry["board_sha256"]:
                continue
            moves = row.get("solution")
            if moves is None:
                moves = row.get("result", {}).get("moves")
            if moves is None:
                continue
            board = CatalogEntry(
                date.fromisoformat(entry["date"]),
                entry["name"],
                entry["width"],
                entry["height"],
                entry["puzzle"],
            ).board()
            if verify_solution(board, moves).ok and (
                board_id not in paths or len(moves) < len(paths[board_id][1])
            ):
                paths[board_id] = (board, moves)
    datasets = {"train": [], "validation": [], "development": []}
    manifest = []
    for index, (board_id, (board, moves)) in enumerate(sorted(paths.items())):
        if a.max_boards is not None and index >= a.max_boards:
            break
        examples = examples_for_solution(
            board_id=board_id, board=board, moves=moves, verify=True
        )
        # The old exporter contains keys for optional memorization. Drop them
        # before fitting or writing any V4 candidate artifact.
        for example in examples:
            example.pop("state_action_key", None)
        datasets[split[board_id]].extend(examples)
        manifest.append(
            {
                "id": board_id,
                "identity": identity(boards[board_id]),
                "split": split[board_id],
                "board_sha256": boards[board_id]["board_sha256"],
                "examples": len(examples),
            }
        )
        print(f"{board_id} {split[board_id]} {len(examples)} examples", flush=True)
    fit = datasets["train"] + datasets["development"]
    if not fit:
        p.error("No verified training paths")
    weights = train_pairwise_ranker(fit, epochs=a.epochs, learning_rate=0.02, seed=4)
    payload = {
        "name": "push-v4-feature-candidate",
        "weights": weights,
        "value_weights": {},
        "state_action_hints": {},
        "priority_recovery": False,
        "experimental": True,
        "training": {
            "seed": 4,
            "epochs": a.epochs,
            "manifest": manifest,
            "validation": ranking_accuracy(datasets["validation"], weights),
            "training": ranking_accuracy(fit, weights),
            "claim": "Held-out ranking accuracy only; search coverage is unmeasured.",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["training"]["validation"]))


if __name__ == "__main__":
    main()
