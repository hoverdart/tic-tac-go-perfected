"""Resolve a dated catalog board (Gemini fallback), solve, and persist it."""

from datetime import UTC, date, datetime
import logging
import traceback
import json
import time
from hashlib import sha256
from typing import Any

from apps.api.board_parser import resolve_daily_board
from apps.api.puzzle_catalog import OFFICIAL_GAME_URL
from apps.api.solution_storage import upsert_solution, begin_daily_solve_run, finish_daily_solve_run, update_daily_solve_run
from apps.api.puzzle_titles import (
    official_level_label,
    title_from_official_catalog,
    title_from_past_days,
)
from solver.service import solve_daily_board, daily_push_version


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
    run_id = begin_daily_solve_run(target_date, daily_push_version())
    run_started = time.perf_counter()
    diagnostics: dict[str, Any] = {}
    board_hash = None
    source_url = OFFICIAL_GAME_URL
    puzzle_title: str | None = None
    board: list[list[str]] | None = None
    parser_name = "not-run"
    solver_name = "not-run"
    logger.info("daily_solve.start puzzle_date=%s", target_date)
    try:
        resolved = resolve_daily_board(target_date)
        board = resolved.board
        board_hash = (
            resolved.board_sha256
            or sha256(json.dumps(board, separators=(",", ":")).encode()).hexdigest()
        )
        diagnostics["board_hash_format"] = (
            "catalog" if resolved.board_sha256 else "wire-json"
        )
        puzzle_title = resolved.puzzle_title or fallback_puzzle_title(target_date)
        parser_name = resolved.parser_name
        source_url = resolved.source_url
        diagnostics["parse_ms"] = (time.perf_counter() - run_started) * 1000
        diagnostics.update(
            phase="push_running", parser_name=parser_name, source_url=source_url
        )
        update_daily_solve_run(run_id, diagnostics=diagnostics, board_sha256=board_hash)
        logger.info(
            "daily_solve.parse.ok parser=%s date=%s source_url=%s board=\n%s",
            parser_name,
            target_date,
            source_url,
            _board_lines(board),
        )

        def persist_push_diagnostics(push_diagnostics):
            diagnostics.update(push_diagnostics)
            update_daily_solve_run(
                run_id, diagnostics=diagnostics, board_sha256=board_hash
            )

        solve_result = solve_daily_board(
            board, on_push_complete=persist_push_diagnostics
        )
        diagnostics.update(solve_result.get("diagnostics", {}))
        solver_name = solve_result["solver_name"]
        logger.info(
            "daily_solve.solve.ok solver=%s solved=%s moves=%r states=%s elapsed_ms=%s",
            solver_name,
            solve_result["solved"],
            solve_result["moves"],
            solve_result["states_checked"],
            solve_result["elapsed_ms"],
        )
    except Exception as exc:
        diagnostics.update(getattr(exc, "diagnostics", {}))
        logger.error(
            "daily_solve.failed date=%s error=%s\n%s",
            target_date,
            exc,
            traceback.format_exc(),
        )
        record = _failed_record(
            target_date,
            source_url,
            str(exc),
            board=board,
            puzzle_title=puzzle_title or fallback_puzzle_title(target_date),
            solver_name=solver_name,
            parser_name=parser_name,
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
            "error_message": None
            if solve_result["solved"]
            else "Solver did not find a solution.",
            "puzzle_title": puzzle_title,
        }
    diagnostics.update(
        parser_name=parser_name,
        source_url=source_url,
        total_wall_ms=(time.perf_counter() - run_started) * 1000,
    )
    # This receipt survives solved-record protection and persistence errors.
    persistence_started = time.perf_counter()
    try:
        stored = upsert_solution(record, preserve_solved=True)
    except Exception as exc:
        diagnostics["requested_status"] = record["status"]
        finish_daily_solve_run(
            run_id,
            status="failed",
            diagnostics=diagnostics,
            board_sha256=board_hash,
            error_message=f"Persistence failed: {exc}",
        )
        raise
    diagnostics["stored_status"] = stored.get("status")
    diagnostics["persistence_ms"] = (time.perf_counter() - persistence_started) * 1000
    diagnostics["total_wall_ms"] = (time.perf_counter() - run_started) * 1000
    finish_daily_solve_run(
        run_id,
        status=record["status"],
        diagnostics=diagnostics,
        board_sha256=board_hash,
        error_message=record.get("error_message"),
    )
    logger.info(
        "daily_solve.end requested_status=%s stored_status=%s date=%s",
        record["status"],
        stored.get("status"),
        target_date,
    )
    return stored
