"""Fixed hard-tail benchmark and classification suite for the push solver."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from solver.push_solver import core
from solver.push_solver.training_export import (
    DEFAULT_BOARDS_PATH,
    DEFAULT_SOLUTIONS_PATH,
    decode_board,
    load_boards,
    load_solutions,
    push_path,
)
from solver.push_solver.verify import verify_solution


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BENCHMARK_CSV = (
    REPO_ROOT
    / "solver"
    / "gymnasium_register"
    / "push_solver_on_heuristic_cnn_failures.csv"
)

HARD_TAIL_BOARD_IDS: tuple[str, ...] = (
    "20250928",
    "20251005",
    "20251116",
    "20251207",
    "20251212",
    "20251221",
    "20251228",
    "20260124",
    "20260208",
    "20260220",
    "20260301",
    "20260314",
    "20260524",
    "20260614",
    "20260627",
    "20260802",
)


@dataclass(frozen=True)
class HardTailRow:
    board_id: str
    title: str
    known_solution_status: str
    known_pushes: int | None
    known_moves: int | None
    current_solved: bool | None
    current_strategy: str | None
    current_failure_reason: str | None
    current_nodes_expanded: int | None
    run_solved: bool | None = None
    run_strategy: str | None = None
    run_push_depth: int | None = None
    run_keystrokes: int | None = None
    run_nodes_expanded: int | None = None
    run_peak_closed_size: int | None = None
    run_elapsed_ms: float | None = None
    run_failure_reason: str | None = None
    run_verified: bool | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "board_id": self.board_id,
            "title": self.title,
            "known_solution_status": self.known_solution_status,
            "known_pushes": "" if self.known_pushes is None else self.known_pushes,
            "known_moves": "" if self.known_moves is None else self.known_moves,
            "current_solved": "" if self.current_solved is None else self.current_solved,
            "current_strategy": self.current_strategy or "",
            "current_failure_reason": self.current_failure_reason or "",
            "current_nodes_expanded": ""
            if self.current_nodes_expanded is None
            else self.current_nodes_expanded,
            "run_solved": "" if self.run_solved is None else self.run_solved,
            "run_strategy": self.run_strategy or "",
            "run_push_depth": "" if self.run_push_depth is None else self.run_push_depth,
            "run_keystrokes": "" if self.run_keystrokes is None else self.run_keystrokes,
            "run_nodes_expanded": ""
            if self.run_nodes_expanded is None
            else self.run_nodes_expanded,
            "run_peak_closed_size": ""
            if self.run_peak_closed_size is None
            else self.run_peak_closed_size,
            "run_elapsed_ms": ""
            if self.run_elapsed_ms is None
            else f"{self.run_elapsed_ms:.1f}",
            "run_failure_reason": self.run_failure_reason or "",
            "run_verified": "" if self.run_verified is None else self.run_verified,
        }


def load_current_rows(path: Path = DEFAULT_BENCHMARK_CSV) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["board_id"]: row
            for row in csv.DictReader(handle)
            if row.get("board_id")
        }


def known_solutions_by_id(path: Path = DEFAULT_SOLUTIONS_PATH) -> dict[str, str | None]:
    return {
        str(row.get("id")): row.get("solution")
        for row in load_solutions(path)
    }


def _bool_from_csv(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _int_from_csv(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return int(text)


def classify_rows(
    *,
    board_ids: Iterable[str] = HARD_TAIL_BOARD_IDS,
    boards_path: Path = DEFAULT_BOARDS_PATH,
    solutions_path: Path = DEFAULT_SOLUTIONS_PATH,
    benchmark_csv: Path = DEFAULT_BENCHMARK_CSV,
) -> list[HardTailRow]:
    boards = load_boards(boards_path)
    solutions = known_solutions_by_id(solutions_path)
    current_rows = load_current_rows(benchmark_csv)
    rows: list[HardTailRow] = []

    for board_id in board_ids:
        entry = boards[board_id]
        solution = solutions.get(board_id)
        known_pushes: int | None = None
        known_moves: int | None = None
        if solution:
            board = decode_board(entry)
            known_pushes = len(push_path(board, str(solution)))
            known_moves = len(str(solution))
            known_status = "known_solution"
        else:
            known_status = "no_known_solution"

        current = current_rows.get(board_id, {})
        rows.append(
            HardTailRow(
                board_id=board_id,
                title=str(entry.get("name", "")),
                known_solution_status=known_status,
                known_pushes=known_pushes,
                known_moves=known_moves,
                current_solved=_bool_from_csv(current.get("solved")),
                current_strategy=current.get("strategy") or None,
                current_failure_reason=current.get("failure_reason") or None,
                current_nodes_expanded=_int_from_csv(current.get("nodes_expanded")),
            )
        )
    return rows


def run_solver_for_row(
    row: HardTailRow,
    *,
    boards: dict[str, dict[str, Any]],
    timeout_seconds: float,
    max_nodes: int,
    weight: float,
) -> HardTailRow:
    board = decode_board(boards[row.board_id])
    started = time.perf_counter()
    result = core.solve(
        board,
        timeout_seconds=timeout_seconds,
        max_nodes=max_nodes,
        weight=weight,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    verified = False
    if result.solved and result.moves is not None:
        verified = verify_solution(board, result.moves).ok

    return HardTailRow(
        **{
            **row.__dict__,
            "run_solved": result.solved,
            "run_strategy": result.strategy,
            "run_push_depth": len(result.pushes) if result.solved else None,
            "run_keystrokes": len(result.moves or "") if result.solved else None,
            "run_nodes_expanded": result.nodes_expanded,
            "run_peak_closed_size": result.peak_closed_size,
            "run_elapsed_ms": elapsed_ms,
            "run_failure_reason": result.failure_reason,
            "run_verified": verified if result.solved else False,
        }
    )


def maybe_run_solver(
    rows: list[HardTailRow],
    *,
    run_solver: bool,
    boards_path: Path,
    timeout_seconds: float,
    max_nodes: int,
    weight: float,
) -> list[HardTailRow]:
    if not run_solver:
        return rows
    boards = load_boards(boards_path)
    return [
        run_solver_for_row(
            row,
            boards=boards,
            timeout_seconds=timeout_seconds,
            max_nodes=max_nodes,
            weight=weight,
        )
        for row in rows
    ]


def print_rows(rows: list[HardTailRow]) -> None:
    fieldnames = list(rows[0].as_dict()) if rows else list(HardTailRow.__annotations__)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-id", action="append", default=[])
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--boards-path", type=Path, default=DEFAULT_BOARDS_PATH)
    parser.add_argument("--solutions-path", type=Path, default=DEFAULT_SOLUTIONS_PATH)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    args = parser.parse_args()

    board_ids = tuple(args.board_id) if args.board_id else HARD_TAIL_BOARD_IDS
    rows = classify_rows(
        board_ids=board_ids,
        boards_path=args.boards_path,
        solutions_path=args.solutions_path,
        benchmark_csv=args.benchmark_csv,
    )
    rows = maybe_run_solver(
        rows,
        run_solver=args.run_solver,
        boards_path=args.boards_path,
        timeout_seconds=args.timeout_seconds,
        max_nodes=args.max_nodes,
        weight=args.weight,
    )
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
