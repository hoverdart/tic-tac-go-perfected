"""Solve every historical board with the production heuristic-CNN solver.

Results are written incrementally as JSON Lines so a long run can be stopped
and resumed without losing completed boards.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_PATH = SCRIPT_DIR / "allBoards.json"
OUTPUT_PATH = SCRIPT_DIR / "all_boards_heuristic_cnn_solutions.jsonl"

WORKERS = 6
RESUME = True

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


def decode_board(entry: dict[str, Any]) -> list[list[str]]:
    """Convert one compact manifest puzzle into the solver's board format."""
    width = int(entry["width"])
    height = int(entry["height"])
    puzzle = str(entry["puzzle"])
    expected_cells = width * height
    if len(puzzle) != expected_cells:
        raise ValueError(
            f"Puzzle {entry.get('id')} has {len(puzzle)} cells; "
            f"expected {expected_cells}."
        )

    return [
        [CELL_MAP[cell] for cell in puzzle[row * width : (row + 1) * width]]
        for row in range(height)
    ]


def solve_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Solve one manifest entry inside a worker process."""
    result = {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "date": entry.get("date"),
        "puzzle": entry.get("puzzle"),
        "solution": None,
    }

    try:
        # Import in the worker so each spawned process owns its model/runtime.
        from solver.heuristic_cnn_solver import solve

        moves, _final_board, _states_checked = solve(decode_board(entry))
        result["solution"] = moves
        result["_error"] = None if moves is not None else "No solution found."
    except Exception as exc:
        result["_error"] = f"{type(exc).__name__}: {exc}"

    return result


def load_entries() -> list[dict[str, Any]]:
    """Load the manifest in release-date order."""
    entries = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected a JSON array in {INPUT_PATH}.")
    return sorted(entries, key=lambda entry: str(entry.get("id", "")))


def completed_ids() -> set[str]:
    """Return board IDs already present in the output file."""
    if not RESUME or not OUTPUT_PATH.exists():
        return set()

    completed: set[str] = set()
    for line_number, line in enumerate(
        OUTPUT_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {OUTPUT_PATH} on line {line_number}."
            ) from exc
        if record.get("id") is not None:
            completed.add(str(record["id"]))
    return completed


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove worker-only diagnostic fields before writing output."""
    return {
        "id": result["id"],
        "name": result["name"],
        "date": result["date"],
        "puzzle": result["puzzle"],
        "solution": result["solution"],
    }


def main() -> int:
    entries = load_entries()
    done = completed_ids()
    pending = [entry for entry in entries if str(entry.get("id")) not in done]

    print(
        f"Boards: {len(entries)} total, {len(done)} completed, "
        f"{len(pending)} pending; workers={WORKERS}",
        flush=True,
    )
    if not pending:
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    solved_count = 0
    failed_count = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as output_file:
        with ProcessPoolExecutor(
            max_workers=WORKERS,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(solve_entry, entry) for entry in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                output_file.write(
                    json.dumps(public_result(result), separators=(",", ":")) + "\n"
                )
                output_file.flush()

                if result["solution"] is None:
                    failed_count += 1
                    outcome = f"failed ({result.get('_error')})"
                else:
                    solved_count += 1
                    outcome = f"solved in {len(result['solution'])} moves"

                print(
                    f"{index}/{len(pending)} id={result['id']} "
                    f"name={result['name']!r}: {outcome}",
                    flush=True,
                )

    print(
        f"Finished: solved={solved_count}, failed={failed_count}, "
        f"output={OUTPUT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
