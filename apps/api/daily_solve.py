"""Resolve a dated catalog board (Gemini fallback), solve, and persist it."""

from datetime import UTC, date, datetime
import logging
import traceback
from typing import Any

from apps.api.board_parser import resolve_daily_board
from apps.api.puzzle_catalog import OFFICIAL_GAME_URL
from apps.api.solution_storage import upsert_solution
from apps.api.puzzle_titles import (
    official_level_label,
    title_from_official_catalog,
    title_from_past_days,
)
from solver.service import solve_daily_board


logger = logging.getLogger("tic_tac_go.daily_solve")


def _board_lines(board: list[list[str]] | None) -> str:
    if not board:
        return "<none>"
    return "\n".join(" ".join(cell if cell else "." for cell in row) for row in board)


def utc_puzzle_date() -> date:
    return datetime.now(UTC).date()


def fallback_puzzle_title(puzzle_date: date) -> str | None:
    return (
        title_from_official_catalog(puzzle_date)
        or title_from_past_days(puzzle_date)
        or official_level_label(puzzle_date)
    )


def _failed_record(
    puzzle_date: date,
    source_url: str,
    error_message: str,
    board: list[list[str]] | None = None,
    puzzle_title: str | None = None,
    solver_name: str = "not-run",
    parser_name: str = "not-run",
) -> dict[str, Any]:
    return {
        "puzzle_date": puzzle_date,
        "source_url": source_url,
        "parser_name": parser_name,
        "solver_name": solver_name,
        "board": board,
        "moves": None,
        "final_board": None,
        "step_boards": [],
        "states_checked": None,
        "elapsed_ms": None,
        "status": "failed",
        "error_message": error_message,
        "puzzle_title": puzzle_title,
    }


def run_daily_solve(puzzle_date: date | None = None) -> dict[str, Any]:
    """Use the requested date throughout; never fall back because search failed.

    No browser or Gemini call is needed when the catalog has a valid board.
    Failure records cannot overwrite an existing solved record, including when
    concurrent jobs finish in the opposite order.
    """
    target_date = puzzle_date or utc_puzzle_date()
    source_url = OFFICIAL_GAME_URL
    puzzle_title: str | None = None
    board: list[list[str]] | None = None
    parser_name = "not-run"
    solver_name = "not-run"
    logger.info("daily_solve.start puzzle_date=%s", target_date)
    try:
        resolved = resolve_daily_board(target_date)
        board = resolved.board
        puzzle_title = resolved.puzzle_title or fallback_puzzle_title(target_date)
        parser_name = resolved.parser_name
        source_url = resolved.source_url
        logger.info("daily_solve.parse.ok parser=%s date=%s source_url=%s board=\n%s",
                    parser_name, target_date, source_url, _board_lines(board))
        solve_result = solve_daily_board(board)
        solver_name = solve_result["solver_name"]
        logger.info("daily_solve.solve.ok solver=%s solved=%s moves=%r states=%s elapsed_ms=%s",
                    solver_name, solve_result["solved"], solve_result["moves"],
                    solve_result["states_checked"], solve_result["elapsed_ms"])
    except Exception as exc:
        logger.error("daily_solve.failed date=%s error=%s\n%s", target_date, exc, traceback.format_exc())
        record = _failed_record(
            target_date, source_url, str(exc), board=board,
            puzzle_title=puzzle_title or fallback_puzzle_title(target_date),
            solver_name=solver_name, parser_name=parser_name,
        )
    else:
        record = {
            "puzzle_date": target_date,
            "source_url": source_url,
            "parser_name": parser_name,
            "solver_name": solver_name,
            "board": solve_result["start_board"],
            "moves": solve_result["moves"],
            "final_board": solve_result["final_board"],
            "step_boards": solve_result["steps"],
            "states_checked": solve_result["states_checked"],
            "elapsed_ms": solve_result["elapsed_ms"],
            "status": "solved" if solve_result["solved"] else "unsolved",
            "error_message": None if solve_result["solved"] else "Solver did not find a solution.",
            "puzzle_title": puzzle_title,
        }
    stored = upsert_solution(record, preserve_solved=True)
    logger.info("daily_solve.end requested_status=%s stored_status=%s date=%s",
                record["status"], stored.get("status"), target_date)
    return stored
