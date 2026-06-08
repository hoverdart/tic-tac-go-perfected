from __future__ import annotations

import argparse
import time

from solver import optimized_solver
from solver.gymnasium_register import ranked_real_boards
from solver.randomPythonFiles import superTicTacGoSolver as legacy_solver


BOARD_GROUPS = {
    "five": ranked_real_boards.fiveBoards,
    "six": ranked_real_boards.sixBoards,
    "seven": ranked_real_boards.sevenBoards,
    "eight": ranked_real_boards.eightBoards,
    "nine": ranked_real_boards.nineBoards,
}


def _run_solver(name, solve_fn, board, max_states):
    started = time.perf_counter()
    moves, _final_board, states = solve_fn(
        board,
        progress_every=0,
        max_states=max_states,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "name": name,
        "solved": moves is not None,
        "moves": None if moves is None else len(moves),
        "states": states,
        "elapsed_ms": elapsed_ms,
    }


def _iter_boards(groups, limit):
    for group in groups:
        boards = BOARD_GROUPS[group]
        for index, board in enumerate(boards[:limit]):
            yield group, index, board


def main():
    parser = argparse.ArgumentParser(
        description="Compare the legacy and optimized Tic-Tac-Go solvers."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(BOARD_GROUPS),
        default=["five", "six", "seven"],
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument(
        "--mode",
        choices=("hybrid", "fast", "exact"),
        default="hybrid",
        help="Optimized solver mode.",
    )
    args = parser.parse_args()

    print("group,index,solver,solved,moves,states,elapsed_ms")
    for group, index, board in _iter_boards(args.groups, args.limit):
        results = [
            _run_solver("legacy", legacy_solver.solve, board, args.max_states),
            _run_solver(
                f"optimized-{args.mode}",
                lambda *solve_args, **solve_kwargs: optimized_solver.solve(
                    *solve_args,
                    mode=args.mode,
                    **solve_kwargs,
                ),
                board,
                args.max_states,
            ),
        ]
        for result in results:
            print(
                ",".join(
                    [
                        group,
                        str(index),
                        result["name"],
                        str(result["solved"]),
                        "" if result["moves"] is None else str(result["moves"]),
                        str(result["states"]),
                        f"{result['elapsed_ms']:.1f}",
                    ]
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
