"""Compare push solver and production heuristic/CNN beam solver on all boards."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BOARDS_PATH = SCRIPT_DIR / "allBoards.json"

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

FIELDNAMES = [
    "board_id",
    "title",
    "push_solved",
    "push_verified",
    "push_strategy",
    "push_moves",
    "push_nodes",
    "push_elapsed_ms",
    "push_failure_reason",
    "beam_solved",
    "beam_verified",
    "beam_moves",
    "beam_states",
    "beam_elapsed_ms",
    "combined_solved",
]


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


def load_entries(path: Path) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    return sorted(entries, key=lambda entry: str(entry.get("id", "")))


def run_push(board, *, timeout_seconds: float, max_nodes: int) -> dict[str, Any]:
    from solver.push_solver import solve, verify_solution

    started = time.perf_counter()
    result = solve(board, timeout_seconds=timeout_seconds, max_nodes=max_nodes)
    elapsed_ms = (time.perf_counter() - started) * 1000
    verification = verify_solution(board, result.moves) if result.solved else None
    return {
        "solved": result.solved,
        "verified": bool(verification and verification.ok),
        "strategy": result.strategy or "",
        "moves": "" if result.moves is None else len(result.moves),
        "nodes": result.nodes_expanded,
        "elapsed_ms": elapsed_ms,
        "failure_reason": result.failure_reason or "",
    }


def run_beam(board, *, timeout_seconds: float, beam_width: int, max_depth: int) -> dict[str, Any]:
    from solver import heuristic_cnn_solver
    from solver.push_solver import verify_solution

    started = time.perf_counter()
    moves, _final_board, states = heuristic_cnn_solver.solve(
        board,
        beam_width=beam_width,
        max_depth=max_depth,
        attempt_timeout_seconds=timeout_seconds,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    verification = verify_solution(board, moves) if moves is not None else None
    return {
        "solved": moves is not None,
        "verified": bool(verification and verification.ok),
        "moves": "" if moves is None else len(moves),
        "states": states,
        "elapsed_ms": elapsed_ms,
    }


def row_for_entry(
    entry: dict[str, Any],
    *,
    run_push_solver: bool,
    run_beam_solver: bool,
    push_timeout_seconds: float,
    push_max_nodes: int,
    beam_timeout_seconds: float,
    beam_width: int,
    beam_max_depth: int,
) -> dict[str, Any]:
    board = decode_board(entry)
    push = (
        run_push(
            board,
            timeout_seconds=push_timeout_seconds,
            max_nodes=push_max_nodes,
        )
        if run_push_solver
        else {
            "solved": False,
            "verified": False,
            "strategy": "",
            "moves": "",
            "nodes": "",
            "elapsed_ms": "",
            "failure_reason": "",
        }
    )
    beam = (
        run_beam(
            board,
            timeout_seconds=beam_timeout_seconds,
            beam_width=beam_width,
            max_depth=beam_max_depth,
        )
        if run_beam_solver
        else {
            "solved": False,
            "verified": False,
            "moves": "",
            "states": "",
            "elapsed_ms": "",
        }
    )
    return {
        "board_id": str(entry.get("id", "")),
        "title": str(entry.get("name", "")),
        "push_solved": push["solved"],
        "push_verified": push["verified"],
        "push_strategy": push["strategy"],
        "push_moves": push["moves"],
        "push_nodes": push["nodes"],
        "push_elapsed_ms": ""
        if push["elapsed_ms"] == ""
        else f"{push['elapsed_ms']:.1f}",
        "push_failure_reason": push["failure_reason"],
        "beam_solved": beam["solved"],
        "beam_verified": beam["verified"],
        "beam_moves": beam["moves"],
        "beam_states": beam["states"],
        "beam_elapsed_ms": ""
        if beam["elapsed_ms"] == ""
        else f"{beam['elapsed_ms']:.1f}",
        "combined_solved": bool(push["verified"] or beam["verified"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    push_solved = {row["board_id"] for row in rows if row["push_verified"] is True}
    beam_solved = {row["board_id"] for row in rows if row["beam_verified"] is True}
    combined = push_solved | beam_solved
    return {
        "total": total,
        "push_solved": len(push_solved),
        "push_accuracy": len(push_solved) / total if total else 0.0,
        "push_accuracy_percent": (len(push_solved) / total * 100.0) if total else 0.0,
        "beam_solved": len(beam_solved),
        "beam_accuracy": len(beam_solved) / total if total else 0.0,
        "beam_accuracy_percent": (len(beam_solved) / total * 100.0) if total else 0.0,
        "combined_solved": len(combined),
        "combined_accuracy": len(combined) / total if total else 0.0,
        "combined_accuracy_percent": (len(combined) / total * 100.0) if total else 0.0,
        "beam_only": sorted(beam_solved - push_solved),
        "push_only": sorted(push_solved - beam_solved),
        "both_failed": sorted(
            {row["board_id"] for row in rows} - combined
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    total = int(summary["total"])
    print(
        "SUMMARY "
        f"push={summary['push_solved']}/{total} "
        f"({summary['push_accuracy_percent']:.2f}%) "
        f"beam={summary['beam_solved']}/{total} "
        f"({summary['beam_accuracy_percent']:.2f}%) "
        f"combined={summary['combined_solved']}/{total} "
        f"({summary['combined_accuracy_percent']:.2f}%)"
    )
    print(
        "OVERLAP "
        f"push_only={len(summary['push_only'])} "
        f"beam_only={len(summary['beam_only'])} "
        f"both_failed={len(summary['both_failed'])}"
    )


def write_csv(rows: list[dict[str, Any]], output: Path | None) -> None:
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=Path, default=DEFAULT_BOARDS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--limit", "--num-tests", dest="limit", type=int, default=None)
    parser.add_argument("--board-id", action="append", default=[])
    parser.add_argument(
        "--solver",
        choices=("both", "push", "beam"),
        default="both",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Default per-solver timeout. Override with solver-specific flags.",
    )
    parser.add_argument("--push-timeout-seconds", type=float, default=None)
    parser.add_argument("--push-max-nodes", type=int, default=500_000)
    parser.add_argument("--beam-timeout-seconds", type=float, default=None)
    parser.add_argument("--beam-width", type=int, default=5_000)
    parser.add_argument("--beam-max-depth", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    entries = load_entries(args.boards)
    if args.board_id:
        wanted = set(args.board_id)
        entries = [entry for entry in entries if str(entry.get("id")) in wanted]
    if args.limit is not None:
        entries = entries[: args.limit]

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    push_timeout_seconds = (
        args.timeout_seconds
        if args.push_timeout_seconds is None
        else args.push_timeout_seconds
    )
    beam_timeout_seconds = (
        args.timeout_seconds
        if args.beam_timeout_seconds is None
        else args.beam_timeout_seconds
    )

    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.workers == 1:
        for index, entry in enumerate(entries, start=1):
            row = row_for_entry(
                entry,
                run_push_solver=args.solver in {"both", "push"},
                run_beam_solver=args.solver in {"both", "beam"},
                push_timeout_seconds=push_timeout_seconds,
                push_max_nodes=args.push_max_nodes,
                beam_timeout_seconds=beam_timeout_seconds,
                beam_width=args.beam_width,
                beam_max_depth=args.beam_max_depth,
            )
            rows_by_id[row["board_id"]] = row
            print(
                f"{index}/{len(entries)} id={row['board_id']} "
                f"push={row['push_verified']} beam={row['beam_verified']} "
                f"combined={row['combined_solved']}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    row_for_entry,
                    entry,
                    run_push_solver=args.solver in {"both", "push"},
                    run_beam_solver=args.solver in {"both", "beam"},
                    push_timeout_seconds=push_timeout_seconds,
                    push_max_nodes=args.push_max_nodes,
                    beam_timeout_seconds=beam_timeout_seconds,
                    beam_width=args.beam_width,
                    beam_max_depth=args.beam_max_depth,
                ): entry
                for entry in entries
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows_by_id[row["board_id"]] = row
                print(
                    f"{index}/{len(entries)} id={row['board_id']} "
                    f"push={row['push_verified']} beam={row['beam_verified']} "
                    f"combined={row['combined_solved']}",
                    flush=True,
                )

    rows = [
        rows_by_id[str(entry.get("id", ""))]
        for entry in entries
        if str(entry.get("id", "")) in rows_by_id
    ]
    write_csv(rows, args.output)
    summary = summarize(rows)
    print_summary(summary)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
