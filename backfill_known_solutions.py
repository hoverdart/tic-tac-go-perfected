"""Backfill Postgres from precomputed heuristic-CNN solution JSONL output."""

from __future__ import annotations

import argparse
from datetime import date
from math import isqrt
import json
import logging
import os
from pathlib import Path
from typing import Any

from apps.api.puzzle_titles import is_generic_title
from apps.api.solution_storage import get_solution, upsert_solution
from solver.board_utils import normalize_board, to_wire_board
from solver.legacy_solver import apply_single_move, solved


LOGGER = logging.getLogger("tic_tac_go.known_solution_backfill")
DEFAULT_INPUT = Path(
    "solver/gymnasium_register/all_boards_heuristic_cnn_solutions.jsonl"
)
DEFAULT_SOURCE_URL = "https://www.google.com/search?q=tic+tac+go&hl=en&gl=US"
PARSER_NAME = "historical-jsonl"
SOLVER_NAME = "heuristic-CNN"

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


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing existing environment values."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def record_date(record: dict[str, Any]) -> date:
    """Parse a JSONL record's date object, falling back to its YYYYMMDD ID."""
    value = record.get("date")
    if isinstance(value, dict):
        return date(int(value["year"]), int(value["month"]), int(value["day"]))

    board_id = str(record.get("id", ""))
    if len(board_id) != 8 or not board_id.isdigit():
        raise ValueError(f"Record has no valid date or YYYYMMDD id: {record!r}")
    return date(int(board_id[:4]), int(board_id[4:6]), int(board_id[6:8]))


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load, validate, deduplicate, and chronologically sort JSONL records."""
    records_by_date: dict[date, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from exc

        puzzle_date = record_date(record)
        if puzzle_date in records_by_date:
            raise ValueError(f"Duplicate puzzle date {puzzle_date} in {path}.")
        records_by_date[puzzle_date] = record

    return [records_by_date[key] for key in sorted(records_by_date)]


def decode_board(puzzle: str) -> tuple[tuple[str, ...], ...]:
    """Decode a compact square puzzle string into the canonical board shape."""
    side = isqrt(len(puzzle))
    if side * side != len(puzzle):
        raise ValueError(f"Puzzle has {len(puzzle)} cells and is not square.")

    board = [
        [CELL_MAP[cell] for cell in puzzle[row * side : (row + 1) * side]]
        for row in range(side)
    ]
    return normalize_board(board)


def replay_solution(
    start_board: tuple[tuple[str, ...], ...],
    moves: str,
) -> tuple[tuple[tuple[str, ...], ...], list[dict[str, Any]]]:
    """Replay and validate a known move sequence while building API step data."""
    board = start_board
    steps: list[dict[str, Any]] = []
    for index, move in enumerate(moves, start=1):
        if move not in {"U", "D", "L", "R"}:
            raise ValueError(f"Invalid move {move!r} at step {index}.")
        next_board = apply_single_move(board, move)
        if next_board == board:
            raise ValueError(f"Move {move!r} at step {index} did not change the board.")
        board = next_board
        steps.append({"move": move, "board": to_wire_board(board)})

    if not solved(board):
        raise ValueError("Move sequence did not produce a solved board.")
    return board, steps


def new_record(
    source: dict[str, Any],
    puzzle_date: date,
) -> dict[str, Any]:
    """Build a complete storage row for a date not currently in Postgres."""
    board = decode_board(str(source["puzzle"]))
    moves = source.get("solution")
    final_board = None
    steps: list[dict[str, Any]] = []
    if moves is not None:
        final_board, steps = replay_solution(board, str(moves))

    return {
        "puzzle_date": puzzle_date,
        "source_url": os.getenv("GOOGLE_TIC_TAC_GO_URL", DEFAULT_SOURCE_URL),
        "parser_name": PARSER_NAME,
        "solver_name": SOLVER_NAME,
        "board": to_wire_board(board),
        "moves": moves,
        "final_board": to_wire_board(final_board) if final_board else None,
        "step_boards": steps,
        "states_checked": None,
        "elapsed_ms": None,
        "status": "solved" if moves is not None else "unsolved",
        "error_message": None if moves is not None else "No solution in import file.",
        "puzzle_title": source.get("name"),
    }


def merge_missing_fields(
    existing: dict[str, Any],
    source: dict[str, Any],
    puzzle_date: date,
) -> tuple[dict[str, Any], list[str]]:
    """Return an updated row containing only source-backed missing information."""
    record = dict(existing)
    changes: list[str] = []

    existing_title = record.get("puzzle_title")
    title_needs_replacement = not existing_title or is_generic_title(existing_title)
    if (
        puzzle_date != date.today()
        and title_needs_replacement
        and source.get("name")
    ):
        record["puzzle_title"] = source["name"]
        changes.append("name")

    source_moves = source.get("solution")
    if not record.get("moves") and source_moves is not None:
        board = decode_board(str(source["puzzle"]))
        final_board, steps = replay_solution(board, str(source_moves))
        record.update(
            {
                "board": to_wire_board(board),
                "moves": source_moves,
                "final_board": to_wire_board(final_board),
                "step_boards": steps,
                "solver_name": SOLVER_NAME,
                "states_checked": None,
                "elapsed_ms": None,
                "status": "solved",
                "error_message": None,
            }
        )
        changes.append("solution")

    return record, changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import precomputed historical solutions into daily_solutions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--include-future",
        action="store_true",
        help="Also import puzzle dates after today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing to Postgres.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    load_env_file(args.env_file)
    records = load_records(args.input)
    if not args.include_future:
        records = [record for record in records if record_date(record) <= date.today()]

    counts = {"inserted": 0, "updated": 0, "complete": 0, "unavailable": 0}
    LOGGER.info("backfill.start records=%s dry_run=%s", len(records), args.dry_run)

    for index, source in enumerate(records, start=1):
        puzzle_date = record_date(source)
        existing = get_solution(puzzle_date)

        if existing is None:
            record = new_record(source, puzzle_date)
            action = "inserted"
            if source.get("solution") is None:
                counts["unavailable"] += 1
        else:
            record, changes = merge_missing_fields(existing, source, puzzle_date)
            if not changes:
                missing_source_solution = (
                    not existing.get("moves") and source.get("solution") is None
                )
                count_key = "unavailable" if missing_source_solution else "complete"
                counts[count_key] += 1
                LOGGER.info(
                    "%s/%s date=%s %s; skipped",
                    index,
                    len(records),
                    puzzle_date,
                    count_key,
                )
                continue
            action = "updated"

        if not args.dry_run:
            upsert_solution(record)
        counts[action] += 1
        LOGGER.info(
            "%s/%s date=%s action=%s title=%r moves=%s",
            index,
            len(records),
            puzzle_date,
            action,
            record.get("puzzle_title"),
            len(record.get("moves") or ""),
        )

    LOGGER.info("backfill.done counts=%s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
