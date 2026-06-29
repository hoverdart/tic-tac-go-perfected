"""
Daily solve job.

Orchestrates the three-step pipeline for each day's Tic Tac Go puzzle:
  1. Capture — screenshot the live Google puzzle via a remote browser.
  2. Parse   — send the screenshot to the Gemini vision parser to extract
               the board grid.
  3. Solve   — run the BFS solver and write the result to Postgres.

Any failure at any step is caught, stored as a "failed" record in the DB,
and re-raised so the caller (the API job endpoint) can surface it properly.
"""

from datetime import UTC, date, datetime
import logging
import traceback
from typing import Any

from apps.api.board_capture import capture_google_board_screenshot, google_tic_tac_go_url
from apps.api.board_parser import PARSER_NAME, parse_board_from_screenshot
from apps.api.solution_storage import upsert_solution
from apps.api.puzzle_titles import title_from_past_days
from solver.service import solve_board


logger = logging.getLogger("tic_tac_go.daily_solve")


def _board_lines(board: list[list[str]] | None) -> str:
    """Format a board grid as a human-readable multi-line string for logging."""
    if not board:
        return "<none>"
    return "\n".join(
        " ".join(cell if cell else "." for cell in row)
        for row in board
    )


def utc_puzzle_date() -> date:
    """Return today's date in UTC — used to key the daily solution record."""
    return datetime.now(UTC).date()


def _failed_record(
    puzzle_date: date,
    source_url: str,
    error_message: str,
    board: list[list[str]] | None = None,
    puzzle_title: str | None = None,
    solver_name: str = "not-run",
) -> dict[str, Any]:
    """Build a minimal 'failed' DB record so we always have something stored."""
    return {
        "puzzle_date": puzzle_date,
        "source_url": source_url,
        "parser_name": PARSER_NAME,
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
    """Run the full capture → parse → solve pipeline for one puzzle date.

    Returns the stored DB record dict. If any step fails the error is
    persisted as a "failed" record and the stored dict is still returned
    (not re-raised) so the job endpoint can respond with meaningful detail.
    """
    target_date = puzzle_date or utc_puzzle_date()
    source_url = google_tic_tac_go_url()

    # Declared here so the except block can safely reference it even if the
    # capture step fails before puzzle_title is assigned.
    puzzle_title: str | None = None
    board: list[list[str]] | None = None
    solver_name = "not-run"

    logger.info("=" * 72)
    logger.info("daily_solve.start puzzle_date=%s source_url=%s", target_date, source_url)

    try:
        # Step 1: capture a screenshot of the live puzzle.
        logger.info("daily_solve.capture.begin")
        capture_result = capture_google_board_screenshot(source_url, puzzle_date=target_date)
        screenshot_path = capture_result.screenshot_path
        puzzle_title = capture_result.puzzle_title
        logger.info(
            "daily_solve.capture.ok screenshot_path=%s puzzle_title=%r",
            screenshot_path,
            puzzle_title,
        )

        # Step 2: send the screenshot to the Gemini parser to extract the grid.
        logger.info("daily_solve.parse.begin screenshot_path=%s", screenshot_path)
        board = parse_board_from_screenshot(screenshot_path)
        logger.info("daily_solve.parse.ok parser=%s board=\n%s", PARSER_NAME, _board_lines(board))

        # Step 3: run the selected solver on the parsed grid.
        logger.info("daily_solve.solve.begin")
        solve_result = solve_board(board)
        solver_name = solve_result["solver_name"]
        logger.info(
            "daily_solve.solve.ok solver=%s solved=%s moves=%r states_checked=%s elapsed_ms=%s",
            solver_name,
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
        if puzzle_title is None:
            puzzle_title = title_from_past_days(target_date)
        record = _failed_record(
            target_date,
            source_url,
            str(exc),
            board=board,
            puzzle_title=puzzle_title,
            solver_name=solver_name,
        )
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
        "solver_name": solve_result["solver_name"],
        "board": solve_result["start_board"],
        "moves": solve_result["moves"],
        "final_board": solve_result["final_board"],
        "step_boards": solve_result["steps"],
        "states_checked": solve_result["states_checked"],
        "elapsed_ms": solve_result["elapsed_ms"],
        "status": status,
        "error_message": None if solve_result["solved"] else "Solver did not find a solution.",
        "puzzle_title": puzzle_title,
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
