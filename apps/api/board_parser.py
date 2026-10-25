"""
Catalog-first board acquisition, with date-checked Gemini screenshot fallback.
"""

from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import date

from apps.api.puzzle_catalog import CatalogError, entry_for_date


PARSER_NAME = "gemini"
logger = logging.getLogger("tic_tac_go.daily_solve")


class DailyBoardParseError(RuntimeError):
    """Raised when the captured board image cannot be parsed."""


@dataclass(frozen=True)
class DailyBoardInput:
    board: list[list[str]]
    puzzle_title: str | None
    parser_name: str
    source_url: str
    board_sha256: str | None = None


def resolve_daily_board(puzzle_date: date) -> DailyBoardInput:
    try:
        entry, snapshot = entry_for_date(puzzle_date)
    except CatalogError as exc:
        logger.warning("parse.catalog_unavailable date=%s error=%s", puzzle_date, exc)
    else:
        logger.info("parse.catalog_ok date=%s board_sha256=%s source_sha256=%s",
                    puzzle_date, entry.board_sha256, snapshot.source_sha256)
        return DailyBoardInput(entry.board(), entry.title, "catalog", snapshot.source_url, entry.board_sha256)

    # The live game may display another date near midnight, or when a historical
    # run was requested. Never relabel that screenshot as the requested puzzle.
    from apps.api.board_capture import capture_google_board_screenshot, google_tic_tac_go_url

    source_url = google_tic_tac_go_url()
    capture = capture_google_board_screenshot(source_url, puzzle_date=puzzle_date)
    if capture.puzzle_date != puzzle_date:
        raise DailyBoardParseError(
            f"Screenshot date {capture.puzzle_date or 'unknown'} does not match {puzzle_date}."
        )
    board = parse_board_from_screenshot(capture.screenshot_path)
    return DailyBoardInput(board, capture.puzzle_title, PARSER_NAME, source_url)


def parse_board_from_screenshot(screenshot_path: Path) -> list[list[str]]:
    """Parse a Tic Tac Go board from a screenshot and return it as a 2-D grid.

    Each cell in the returned grid is one of: "", "X", "O", "U", or "B".
    Raises DailyBoardParseError if the parser can't be loaded or the image
    can't be interpreted.
    """
    # The Gemini parser lives inside the solver package and uses google-genai.
    # Import it here rather than at module top so the
    # rest of the API stays importable in environments where that SDK isn't
    # installed (e.g. running only the solver locally).
    try:
        from solver.boardParsers.fallbackBoardParser import parse_board_from_image
    except ImportError as exc:
        raise DailyBoardParseError("Could not load the Gemini board parser.") from exc

    try:
        logger.info("parse.start parser=%s screenshot_path=%s", PARSER_NAME, screenshot_path)
        return parse_board_from_image(screenshot_path)
    except Exception as exc:
        raise DailyBoardParseError(f"Could not parse Tic Tac Go board: {exc}") from exc
