"""Benchmark the push-level solver on historical Tic Tac Go boards."""

from __future__ import annotations

import argparse
import csv
import sys

from backfill_solutions import ALL_PAST_DAYS, board_from_entry
from solver.push_solver import solve, verify_solution


FIELDNAMES = [
    "board_id",
    "title",
    "solved",
    "push_depth",
    "keystrokes",
    "nodes_expanded",
    "peak_closed_size",
    "elapsed_ms",
    "weight",
    "failure_reason",
    "verified",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the classical push solver.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    entries = ALL_PAST_DAYS[: args.limit] if args.limit is not None else ALL_PAST_DAYS
    for entry in entries:
        board = board_from_entry(entry)
        result = solve(
            board,
            weight=args.weight,
            max_nodes=args.max_nodes,
            timeout_seconds=args.timeout_seconds,
        )
        verified = False
        if result.solved:
            verification = verify_solution(board, result.moves)
            if not verification.ok:
                raise RuntimeError(
                    f"{entry.get('id')} solver returned invalid solution: {verification.error}"
                )
            verified = True
        writer.writerow(
            {
                "board_id": entry.get("id", ""),
                "title": entry.get("name", ""),
                "solved": result.solved,
                "push_depth": len(result.pushes) if result.solved else "",
                "keystrokes": "" if result.moves is None else len(result.moves),
                "nodes_expanded": result.nodes_expanded,
                "peak_closed_size": result.peak_closed_size,
                "elapsed_ms": f"{result.elapsed_ms:.1f}",
                "weight": args.weight,
                "failure_reason": result.failure_reason or "",
                "verified": verified,
            }
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
