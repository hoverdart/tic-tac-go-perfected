"""Compare push solver, production Beam/CNN solver, and their fallback union.

Examples:

    python3 -m solver.gymnasium_register.benchmark_push_vs_beam --timeout-seconds 60 --num-tests 100 --workers 6 --output debug-artifacts/push_vs_beam_100.csv --summary-output debug-artifacts/push_vs_beam_100.json

    python3 -m solver.gymnasium_register.benchmark_push_vs_beam --solver push --timeout-seconds 30 --num-tests 25
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import random
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
    "push_baseline_moves",
    "push_quality_improved",
    "push_quality_nodes",
    "push_nodes",
    "push_elapsed_ms",
    "push_failure_reason",
    "beam_solved",
    "beam_verified",
    "beam_moves",
    "beam_states",
    "beam_elapsed_ms",
    "beam_failure_reason",
    "combined_solved",
    "combined_solver",
    "combined_elapsed_ms",
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
        "baseline_moves": ""
        if result.baseline_keystrokes is None
        else result.baseline_keystrokes,
        "quality_improved": result.quality_improved,
        "quality_nodes": result.quality_nodes_expanded,
        "nodes": result.nodes_expanded,
        "elapsed_ms": elapsed_ms,
        "failure_reason": result.failure_reason or "",
    }


def run_beam(board, *, timeout_seconds: float, beam_width: int, max_depth: int) -> dict[str, Any]:
    from solver.push_solver import verify_solution

    started = time.perf_counter()
    try:
        from solver import heuristic_cnn_solver
    except ModuleNotFoundError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "solved": False,
            "verified": False,
            "moves": "",
            "states": 0,
            "elapsed_ms": elapsed_ms,
            "failure_reason": f"missing_dependency:{exc.name}",
        }

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
        "failure_reason": "" if moves is not None else "not_solved",
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
    fallback_order: str,
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
            "baseline_moves": "",
            "quality_improved": False,
            "quality_nodes": "",
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
            "failure_reason": "",
        }
    )
    if fallback_order == "push-first":
        combined_solver = (
            "push" if push["verified"] else "beam" if beam["verified"] else ""
        )
        combined_elapsed_ms = _sum_elapsed_for_fallback(
            primary=push,
            fallback=beam,
            primary_ran=run_push_solver,
            fallback_ran=run_beam_solver,
        )
    else:
        combined_solver = (
            "beam" if beam["verified"] else "push" if push["verified"] else ""
        )
        combined_elapsed_ms = _sum_elapsed_for_fallback(
            primary=beam,
            fallback=push,
            primary_ran=run_beam_solver,
            fallback_ran=run_push_solver,
        )
    return {
        "board_id": str(entry.get("id", "")),
        "title": str(entry.get("name", "")),
        "push_solved": push["solved"],
        "push_verified": push["verified"],
        "push_strategy": push["strategy"],
        "push_moves": push["moves"],
        "push_baseline_moves": push["baseline_moves"],
        "push_quality_improved": push["quality_improved"],
        "push_quality_nodes": push["quality_nodes"],
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
        "beam_failure_reason": beam["failure_reason"],
        "combined_solved": bool(push["verified"] or beam["verified"]),
        "combined_solver": combined_solver,
        "combined_elapsed_ms": ""
        if combined_elapsed_ms is None
        else f"{combined_elapsed_ms:.1f}",
    }


def _sum_elapsed_for_fallback(
    *,
    primary: dict[str, Any],
    fallback: dict[str, Any],
    primary_ran: bool,
    fallback_ran: bool,
) -> float | None:
    """Estimate wall-clock cost for a sequential combined fallback solver."""
    if not primary_ran:
        if fallback_ran and fallback["elapsed_ms"] != "":
            return fallback["elapsed_ms"]
        return None
    if primary["elapsed_ms"] == "":
        return None
    elapsed_ms = float(primary["elapsed_ms"])
    if primary["verified"] or not fallback_ran:
        return elapsed_ms
    if fallback["elapsed_ms"] != "":
        elapsed_ms += float(fallback["elapsed_ms"])
    return elapsed_ms


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    push_solved = {row["board_id"] for row in rows if row["push_verified"] is True}
    beam_solved = {row["board_id"] for row in rows if row["beam_verified"] is True}
    all_ids = {row["board_id"] for row in rows}
    combined = push_solved | beam_solved
    quality_rows = [
        row
        for row in rows
        if row.get("push_baseline_moves") not in {"", None}
        and row.get("push_moves") not in {"", None}
    ]
    baseline_moves = sum(int(row["push_baseline_moves"]) for row in quality_rows)
    final_moves = sum(int(row["push_moves"]) for row in quality_rows)
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
        "both_solved": len(push_solved & beam_solved),
        "push_only_count": len(push_solved - beam_solved),
        "beam_only_count": len(beam_solved - push_solved),
        "both_failed_count": len(all_ids - combined),
        "push_failed": sorted(all_ids - push_solved),
        "beam_failed": sorted(all_ids - beam_solved),
        "combined_failed": sorted(all_ids - combined),
        "beam_only": sorted(beam_solved - push_solved),
        "push_only": sorted(push_solved - beam_solved),
        "both_failed": sorted(all_ids - combined),
        "quality_eligible": len(quality_rows),
        "quality_improved": sum(
            row.get("push_quality_improved") is True for row in quality_rows
        ),
        "baseline_keystrokes": baseline_moves,
        "final_keystrokes": final_moves,
        "keystrokes_saved": baseline_moves - final_moves,
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
        f"both_solved={summary['both_solved']} "
        f"push_only={summary['push_only_count']} "
        f"beam_only={summary['beam_only_count']} "
        f"both_failed={summary['both_failed_count']}"
    )
    print(
        "QUALITY "
        f"improved={summary['quality_improved']}/{summary['quality_eligible']} "
        f"keystrokes_saved={summary['keystrokes_saved']}"
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--boards", type=Path, default=DEFAULT_BOARDS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--limit",
        "--num-tests",
        dest="limit",
        type=int,
        default=None,
        help="Number of boards to run after board-id and selection filters.",
    )
    parser.add_argument("--board-id", action="append", default=[])
    parser.add_argument(
        "--selection",
        choices=("first", "random"),
        default="first",
        help="How to choose boards when --num-tests/--limit is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for --selection random.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many sorted boards before applying --num-tests.",
    )
    parser.add_argument(
        "--solver",
        choices=("both", "all", "push", "beam"),
        default="both",
        help="'both'/'all' runs both solvers and reports combined fallback accuracy.",
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
    parser.add_argument(
        "--fallback-order",
        choices=("push-first", "beam-first"),
        default="push-first",
        help="Order used to estimate combined fallback solver and elapsed time.",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    entries = load_entries(args.boards)
    if args.board_id:
        wanted = set(args.board_id)
        entries = [entry for entry in entries if str(entry.get("id")) in wanted]
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.start_index:
        entries = entries[args.start_index :]
    if args.selection == "random":
        rng = random.Random(args.seed)
        entries = entries[:]
        rng.shuffle(entries)
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--num-tests/--limit must be non-negative")
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
    run_push_solver = args.solver in {"both", "all", "push"}
    run_beam_solver = args.solver in {"both", "all", "beam"}

    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.workers == 1:
        for index, entry in enumerate(entries, start=1):
            row = row_for_entry(
                entry,
                run_push_solver=run_push_solver,
                run_beam_solver=run_beam_solver,
                push_timeout_seconds=push_timeout_seconds,
                push_max_nodes=args.push_max_nodes,
                beam_timeout_seconds=beam_timeout_seconds,
                beam_width=args.beam_width,
                beam_max_depth=args.beam_max_depth,
                fallback_order=args.fallback_order,
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
                    run_push_solver=run_push_solver,
                    run_beam_solver=run_beam_solver,
                    push_timeout_seconds=push_timeout_seconds,
                    push_max_nodes=args.push_max_nodes,
                    beam_timeout_seconds=beam_timeout_seconds,
                    beam_width=args.beam_width,
                    beam_max_depth=args.beam_max_depth,
                    fallback_order=args.fallback_order,
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
