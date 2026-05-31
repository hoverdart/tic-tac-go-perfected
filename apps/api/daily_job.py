"""
    Defines the daily cron job that captures the 
    current day's Tic Tac Go puzzle, parses it, 
    solves it, and stores the results in the database.
"""


from datetime import UTC, date, datetime
import logging
import traceback
from typing import Any

from apps.api.acquisition import capture_google_board_screenshot, google_tic_tac_go_url
from apps.api.parser import PARSER_NAME, parse_board_from_screenshot
from apps.api.storage import upsert_solution
from solver.service import SOLVER_NAME, solve_board


logger = logging.getLogger("tic_tac_go.daily_solve")


def _board_lines(board: list[list[str]] | None) -> str:
    if not board:
        return "<none>"
    return "\n".join(
        " ".join(cell if cell else "." for cell in row)
        for row in board
    )


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

    logger.info("=" * 72)
    logger.info("daily_solve.start puzzle_date=%s source_url=%s", target_date, source_url)

    try:
        logger.info("daily_solve.capture.begin")
        screenshot_path = capture_google_board_screenshot(source_url)
        logger.info("daily_solve.capture.ok screenshot_path=%s", screenshot_path)

        logger.info("daily_solve.parse.begin screenshot_path=%s", screenshot_path)
        board = parse_board_from_screenshot(screenshot_path)
        logger.info("daily_solve.parse.ok parser=%s board=\n%s", PARSER_NAME, _board_lines(board))

        logger.info("daily_solve.solve.begin solver=%s", SOLVER_NAME)
        solve_result = solve_board(board)
        logger.info(
            "daily_solve.solve.ok solved=%s moves=%r states_checked=%s elapsed_ms=%s",
            solve_result["solved"],
            solve_result["moves"],
            solve_result["states_checked"],
            solve_result["elapsed_ms"],
        )
        logger.info("daily_solve.solve.start_board=\n%s", _board_lines(solve_result["start_board"]))
        logger.info("daily_solve.solve.final_board=\n%s", _board_lines(solve_result["final_board"]))
        logger.info("daily_solve.solve.steps=%s", len(solve_result["steps"]))
    except Exception as exc:
        logger.error("daily_solve.failed error=%s", exc)
        logger.error("daily_solve.traceback\n%s", traceback.format_exc())
        record = _failed_record(target_date, source_url, str(exc))
        logger.info("daily_solve.persist_failed.begin record=%s", record)
        stored = upsert_solution(record)
        logger.info("daily_solve.persist_failed.done stored=%s", stored)
        logger.info("daily_solve.end status=failed")
        logger.info("=" * 72)
        return stored

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
    logger.info(
        "daily_solve.persist.begin status=%s moves=%r states_checked=%s elapsed_ms=%s",
        record["status"],
        record["moves"],
        record["states_checked"],
        record["elapsed_ms"],
    )
    stored = upsert_solution(record)
    logger.info("daily_solve.persist.done stored_status=%s stored_date=%s", stored.get("status"), stored.get("puzzle_date"))
    logger.info("daily_solve.end status=%s", status)
    logger.info("=" * 72)
    return stored
