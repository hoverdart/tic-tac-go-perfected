"""
    Defines the daily cron job that captures the 
    current day's Tic Tac Go puzzle, parses it, 
    solves it, and stores the results in the database.
"""


from datetime import UTC, date, datetime
from typing import Any

from apps.api.acquisition import capture_google_board_screenshot, google_tic_tac_go_url
from apps.api.parser import PARSER_NAME, parse_board_from_screenshot
from apps.api.storage import upsert_solution
from solver.service import SOLVER_NAME, solve_board


def utc_puzzle_date() -> date:
    return datetime.now(UTC).date()


def _failed_record(
    puzzle_date: date,
    source_url: str,
    error_message: str,
    board: list[list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "puzzle_date": puzzle_date,
        "source_url": source_url,
        "parser_name": PARSER_NAME,
        "solver_name": SOLVER_NAME,
        "board": board,
        "moves": None,
        "final_board": None,
        "step_boards": [],
        "states_checked": None,
        "elapsed_ms": None,
        "status": "failed",
        "error_message": error_message,
    }


def run_daily_solve(puzzle_date: date | None = None) -> dict[str, Any]:
    target_date = puzzle_date or utc_puzzle_date()
    source_url = google_tic_tac_go_url()

    try:
        screenshot_path = capture_google_board_screenshot(source_url)
        board = parse_board_from_screenshot(screenshot_path)
        solve_result = solve_board(board)
    except Exception as exc:
        return upsert_solution(_failed_record(target_date, source_url, str(exc)))

    status = "solved" if solve_result["solved"] else "unsolved"
    record = {
        "puzzle_date": target_date,
        "source_url": source_url,
        "parser_name": PARSER_NAME,
        "solver_name": SOLVER_NAME,
        "board": solve_result["start_board"],
        "moves": solve_result["moves"],
        "final_board": solve_result["final_board"],
        "step_boards": solve_result["steps"],
        "states_checked": solve_result["states_checked"],
        "elapsed_ms": solve_result["elapsed_ms"],
        "status": status,
        "error_message": None if solve_result["solved"] else "Solver did not find a solution.",
    }
    return upsert_solution(record)
