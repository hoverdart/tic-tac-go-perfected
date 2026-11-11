"""Independent unit-step oracle, generated puzzles, symmetries and interruptions.

PYTHONPATH=. .venv/bin/python scripts/stress_optimized_paths.py --output-dir /tmp/path-stress
The oracle uses the legacy board replayer, not the new bitmask transition code.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, deque
from hashlib import sha256
from pathlib import Path

from solver.board_utils import normalize_board
from solver.legacy_solver import apply_single_move, lost_check, solved
from solver.optimized_paths import optimize
from solver.optimized_paths.bank import ROOT, load_bank
from solver.optimized_paths.benchmark import engine_sources
from solver.optimized_paths.lower_bound import build_pattern_database
from solver.optimized_paths.model import Budget, Geometry, LimitReached
from solver.optimized_paths.search import bounded_astar
from solver.optimized_paths.verify_benchmark import verify_cohort


def unit_bfs(board, *, max_nodes, goals=1):
    """First popped goal is shortest; later goals provide nonoptimal seeds."""
    start = normalize_board(board)
    if lost_check(start):
        return {"status": "initial_x_loss", "expanded": 0, "answers": []}
    queue = deque([(start, "")])
    seen = {start}
    answers = []
    expanded = 0
    while queue and expanded < max_nodes:
        current, moves = queue.popleft()
        expanded += 1
        if solved(current):
            answers.append(moves)
            if len(answers) >= goals:
                break
            continue  # A real attempt ends at its first win.
        for move in "UDLR":
            child = apply_single_move(current, move)
            if child not in seen and not lost_check(child):
                seen.add(child)
                queue.append((child, moves + move))
    return {
        "status": "solved" if answers else "node_cap" if queue else "unsolvable",
        "expanded": expanded,
        "answers": answers,
    }


def symmetries(board, moves):
    current = tuple(map(tuple, board))
    for _ in range(4):
        yield current, moves
        yield (
            tuple(tuple(reversed(row)) for row in current),
            moves.translate(str.maketrans({"L": "R", "R": "L"})),
        )
        current = tuple(tuple(row) for row in zip(*current[::-1]))
        moves = moves.translate(str.maketrans({"U": "R", "R": "D", "D": "L", "L": "U"}))


class CheckpointBudget:
    """Deterministically interrupt at a check rather than a noisy timer."""

    def __init__(self, stop):
        self.stop = stop
        self.calls = self.expanded = 0

    def check(self):
        self.calls += 1
        if self.calls >= self.stop:
            raise LimitReached("injected_checkpoint")

    def expand(self):
        self.check()
        self.expanded += 1


def require_result(result, optimum):
    if not result.lower_bound <= optimum <= len(result.moves):
        raise AssertionError(
            f"Invalid bound interval: {result.to_dict()}, optimum={optimum}"
        )
    if result.optimality_proven and len(result.moves) != optimum:
        raise AssertionError("False global optimality claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--oracle-nodes", type=int, default=20_000)
    parser.add_argument("--max-xs", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    if bool(args.cohort) != bool(args.manifest):
        parser.error("--cohort and --manifest must be supplied together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(args.seed)
    counters = Counter()
    started = time.perf_counter()
    fixtures = []
    with (args.output_dir / "generated.jsonl").open("w") as output:
        for index in range(args.generated):
            height, width = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 4)][index % 5]
            o_count = 2 if index % 4 else 3
            x_count = index % (args.max_xs + 1)
            wall_count = (index // 3) % 3
            board = [[""] * width for _ in range(height)]
            positions = randomizer.sample(
                range(height * width), 1 + o_count + x_count + wall_count
            )
            for cell, value in zip(
                positions,
                ["U"] + ["O"] * o_count + ["X"] * x_count + ["B"] * wall_count,
            ):
                board[cell // width][cell % width] = value
            oracle = unit_bfs(board, max_nodes=args.oracle_nodes, goals=8)
            row = {"index": index, "board": board, "oracle": oracle}
            counters["generated_" + oracle["status"]] += 1
            if oracle["answers"]:
                optimum = len(oracle["answers"][0])
                seed_path = max(oracle["answers"], key=len)
                counters["nonoptimal_seeds"] += len(seed_path) > optimum
                geometry = Geometry(board)
                pdb = build_pattern_database(
                    geometry, Budget(time.perf_counter() + 1, 50_000)
                )
                if pdb.estimate(geometry.start) > optimum:
                    raise AssertionError("Relaxed database overestimates")
                checks = []
                for cap in (0, 1, 32, 20_000):
                    result = optimize(
                        board, [seed_path], mode="exact", seconds=0.1, max_nodes=cap
                    )
                    require_result(result, optimum)
                    checks.append(
                        {
                            "node_cap": cap,
                            "moves": len(result.moves),
                            "lower_bound": result.lower_bound,
                            "proven": result.optimality_proven,
                        }
                    )
                    counters["generated_optimizer_checks"] += 1
                row["checks"] = checks
                if len(fixtures) < 24 and len(seed_path) > optimum:
                    fixtures.append((board, seed_path, optimum))
                if index % 11 == 0:
                    for transformed, transformed_seed in symmetries(board, seed_path):
                        result = optimize(
                            transformed,
                            [transformed_seed],
                            mode="exact",
                            seconds=0.1,
                            max_nodes=20_000,
                        )
                        require_result(result, optimum)
                        counters["symmetry_checks"] += 1
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
            output.flush()
    interruptions = []
    for index, (board, moves, optimum) in enumerate(fixtures):
        geometry = Geometry(board)
        pdb = build_pattern_database(geometry, Budget(time.perf_counter() + 2, 50_000))
        for stop in range(1, 160):
            path, lower, reason, _states = bounded_astar(
                geometry, moves, CheckpointBudget(stop), pdb=pdb
            )
            if not lower <= optimum <= len(path):
                raise AssertionError(
                    f"Interrupted lower bound invalid: {index}, {stop}, {lower}, {optimum}"
                )
            if reason in ("exhausted", "bound_met") and len(path) != optimum:
                raise AssertionError("Interrupted search falsely certifies a path")
            counters["checkpoint_checks"] += 1
        interruptions.append(
            {
                "fixture": index,
                "board": board,
                "seed": moves,
                "minimum_moves": optimum,
                "checkpoints": 159,
            }
        )
    (args.output_dir / "interruptions.json").write_text(
        json.dumps(interruptions, indent=2) + "\n"
    )
    corpus_checks = []
    if args.cohort:
        verify_cohort(args.cohort, args.manifest)
        bank = load_bank(args.manifest)
        rows = [
            json.loads(line)
            for line in (args.cohort / "results.jsonl").read_text().splitlines()
        ]
        for row in sorted(
            rows,
            key=lambda item: (item.get("result", {}).get("move_count", 0), item["id"]),
        ):
            if not row.get("result", {}).get("optimality_proven"):
                continue
            oracle = unit_bfs(bank.cases[row["id"]].board, max_nodes=args.oracle_nodes)
            status = "inconclusive_node_cap"
            if oracle["answers"]:
                if len(oracle["answers"][0]) != row["result"]["move_count"]:
                    raise AssertionError(
                        f"Corpus shortest-path claim differs: {row['id']}"
                    )
                status = "independently_confirmed"
            elif oracle["status"] != "node_cap":
                raise AssertionError(
                    "Verified seed is not reachable in independent oracle"
                )
            counters["corpus_" + status] += 1
            corpus_checks.append(
                {
                    "id": row["id"],
                    "moves": row["result"]["move_count"],
                    "status": status,
                    "oracle_expanded": oracle["expanded"],
                }
            )
        (args.output_dir / "corpus-oracle.json").write_text(
            json.dumps(corpus_checks, indent=2) + "\n"
        )
    summary = {
        "scope": "independent_correctness_stress_not_timing_benchmark",
        "seed": args.seed,
        "generated_requested": args.generated,
        "oracle_node_cap": args.oracle_nodes,
        "max_xs": args.max_xs,
        "counts": dict(counters),
        "elapsed_seconds": time.perf_counter() - started,
        "engine_sources": engine_sources(),
        "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle_source_sha256": sha256(
            (ROOT / "solver/legacy_solver.py").read_bytes()
        ).hexdigest(),
        "all_checks_passed": True,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
