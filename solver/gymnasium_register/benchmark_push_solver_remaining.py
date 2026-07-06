"""Benchmark historical boards with the push solver.

Normal mode reruns every board. ``--only-failed`` retries only rows currently
marked unsolved. Each completed result replaces its prior row and is written
atomically back to the same file in release order.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_PATH = SCRIPT_DIR / "allBoards.json"
OUTPUT_PATH = SCRIPT_DIR / "push_solver_on_heuristic_cnn_failures.csv"

# Set True to rerun only rows currently marked failed in OUTPUT_PATH.
# Set False to rerun every board in allBoards.json.
ONLY_FAILED = False

FIELDNAMES = [
    "board_id",
    "title",
    "solved",
    "push_depth",
    "keystrokes",
    "nodes_expanded",
    "peak_closed_size",
    "elapsed_ms",
    "failure_reason",
    "verified",
]

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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.push_solver import solve, verify_solution  # noqa: E402


class HardTimeout(Exception):
    """Raised when one board exceeds its wall-clock budget."""


_timeout_active = False


def _timeout_handler(_signum, _frame) -> None:
    if not _timeout_active:
        return
    # Disarm before raising so cleanup cannot be interrupted by a second alarm.
    signal.setitimer(signal.ITIMER_REAL, 0)
    raise HardTimeout


def decode_board(entry: dict[str, Any]) -> list[list[str]]:
    """Decode one compact allBoards.json puzzle."""
    width = int(entry["width"])
    height = int(entry["height"])
    puzzle = str(entry["puzzle"])
    if len(puzzle) != width * height:
        raise ValueError(
            f"Puzzle {entry.get('id')} has {len(puzzle)} cells; "
            f"expected {width * height}."
        )
    return [
        [CELL_MAP[cell] for cell in puzzle[row * width : (row + 1) * width]]
        for row in range(height)
    ]


def load_entries() -> list[dict[str, Any]]:
    entries = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected a JSON array in {INPUT_PATH}.")
    return sorted(entries, key=lambda entry: str(entry.get("id", "")))


def load_existing_rows() -> dict[str, dict[str, str]]:
    if not OUTPUT_PATH.exists():
        return {}
    with OUTPUT_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            row["board_id"]: row
            for row in csv.DictReader(handle)
            if row.get("board_id")
        }


def row_is_solved(row: dict[str, Any]) -> bool:
    return str(row.get("solved", "")).strip().lower() == "true"


def select_pending_entries(
    entries: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    *,
    only_failed: bool,
) -> list[dict[str, Any]]:
    """Select failed existing rows for retry, or every board in normal mode."""
    if only_failed:
        return [
            entry
            for entry in entries
            if str(entry["id"]) in rows_by_id
            and not row_is_solved(rows_by_id[str(entry["id"])])
        ]
    return entries


def write_rows(
    rows_by_id: dict[str, dict[str, Any]],
    ordered_ids: list[str],
) -> None:
    """Atomically persist all completed rows in release order."""
    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for board_id in ordered_ids:
            row = rows_by_id.get(board_id)
            if row is not None:
                writer.writerow(row)
    os.replace(temporary_path, OUTPUT_PATH)


def solve_entry(
    entry: dict[str, Any],
    *,
    weight: float,
    max_nodes: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    global _timeout_active
    board = decode_board(entry)
    started = time.perf_counter()
    try:
        _timeout_active = True
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        result = solve(
            board,
            weight=weight,
            max_nodes=max_nodes,
            timeout_seconds=timeout_seconds,
        )
    except HardTimeout:
        return {
            "board_id": entry["id"],
            "title": entry.get("name", ""),
            "solved": False,
            "push_depth": "",
            "keystrokes": "",
            "nodes_expanded": "",
            "peak_closed_size": "",
            "elapsed_ms": f"{(time.perf_counter() - started) * 1000:.1f}",
            "failure_reason": "hard_timeout",
            "verified": False,
        }
    finally:
        _timeout_active = False
        signal.setitimer(signal.ITIMER_REAL, 0)

    verification = verify_solution(board, result.moves) if result.solved else None
    verified = bool(verification and verification.ok)
    if result.solved and not verified:
        raise RuntimeError(
            f"{entry['id']} solver returned an invalid solution: "
            f"{verification.error if verification else 'not verified'}"
        )

    return {
        "board_id": entry["id"],
        "title": entry.get("name", ""),
        "solved": result.solved,
        "push_depth": len(result.pushes) if result.solved else "",
        "keystrokes": len(result.moves) if result.moves is not None else "",
        "nodes_expanded": result.nodes_expanded,
        "peak_closed_size": result.peak_closed_size,
        "elapsed_ms": f"{result.elapsed_ms:.1f}",
        "failure_reason": result.failure_reason or "",
        "verified": verified,
    }


def solve_entry_worker(
    entry: dict[str, Any],
    weight: float,
    max_nodes: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Initialize worker-local timeout handling and solve one board."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    return solve_entry(
        entry,
        weight=weight,
        max_nodes=max_nodes,
        timeout_seconds=timeout_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the push solver benchmark on all boards or failed CSV rows."
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--only-failed",
        action=argparse.BooleanOptionalAction,
        default=ONLY_FAILED,
        help="Rerun and replace failed CSV rows instead of processing missing boards.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    entries = load_entries()
    ordered_ids = [str(entry["id"]) for entry in entries]
    rows_by_id = load_existing_rows()
    pending = select_pending_entries(
        entries,
        rows_by_id,
        only_failed=args.only_failed,
    )
    mode = "failed rows" if args.only_failed else "all boards"
    print(
        f"Boards={len(entries)} existing={len(rows_by_id)} pending={len(pending)} "
        f"mode={mode} timeout={args.timeout_seconds}s workers={args.workers}",
        flush=True,
    )

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                solve_entry_worker,
                entry,
                args.weight,
                args.max_nodes,
                args.timeout_seconds,
            ): entry
            for entry in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            entry = futures[future]
            row = future.result()
            rows_by_id[str(entry["id"])] = row
            write_rows(rows_by_id, ordered_ids)
            print(
                f"{index}/{len(pending)} id={entry['id']} title={entry.get('name')!r} "
                f"solved={row['solved']} moves={row['keystrokes']} "
                f"nodes={row['nodes_expanded']} elapsed={float(row['elapsed_ms']) / 1000:.3f}s "
                f"reason={row['failure_reason'] or '-'} verified={row['verified']}",
                flush=True,
            )

    solved_count = sum(row_is_solved(row) for row in rows_by_id.values())
    print(
        f"Finished solved={solved_count}/{len(rows_by_id)} output={OUTPUT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
