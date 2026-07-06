"""Benchmark Linear Tree Solver V1 against optimized A*."""

from __future__ import annotations

import argparse
import csv
import sys
import time

from solver import optimized_solver
from solver.gymnasium_register import ranked_real_boards
from solver.learned_search import solver as learned_solver


BOARD_GROUPS = {
    "five": ranked_real_boards.fiveBoards,
    "six": ranked_real_boards.sixBoards,
    "seven": ranked_real_boards.sevenBoards,
    "eight": ranked_real_boards.eightBoards,
    "nine": ranked_real_boards.nineBoards,
}


def _run(name, solve_fn, board, max_states, mode):
    started = time.perf_counter()
    moves, _final_board, states = solve_fn(
        board,
        progress_every=0,
        max_states=max_states,
        mode=mode,
    )
    return {
        "solver": name,
        "solved": moves is not None,
        "moves": "" if moves is None else len(moves),
        "states": states,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Linear Tree Solver V1 and optimized A*."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(BOARD_GROUPS),
        default=["five", "six", "seven"],
    )
    parser.add_argument("--limit-per-group", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=50_000)
    parser.add_argument("--mode", choices=("hybrid", "fast", "exact"), default="hybrid")
    args = parser.parse_args()

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "group",
            "index",
            "solver",
            "solved",
            "moves",
            "states",
            "elapsed_ms",
        ],
    )
    writer.writeheader()
    for group in args.groups:
        for index, board in enumerate(BOARD_GROUPS[group][: args.limit_per_group]):
            for result in (
                _run(
                    "optimized",
                    optimized_solver.solve,
                    board,
                    args.max_states,
                    args.mode,
                ),
                _run(
                    "linear-tree-v1",
                    learned_solver.solve,
                    board,
                    args.max_states,
                    args.mode,
                ),
            ):
                writer.writerow(
                    {
                        "group": group,
                        "index": index,
                        "solver": result["solver"],
                        "solved": result["solved"],
                        "moves": result["moves"],
                        "states": result["states"],
                        "elapsed_ms": f"{result['elapsed_ms']:.1f}",
                    }
                )
                sys.stdout.flush()


if __name__ == "__main__":
    main()
