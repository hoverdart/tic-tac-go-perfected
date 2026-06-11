"""
Board parsing layer.

Wraps the Gemini-based vision parser (solver/boardParsers/fallbackBoardParser)
in a thin adapter so the rest of the API only depends on this module rather
than on the solver package directly. Raises `DailyBoardParseError` on any
failure so callers get a single, consistent exception type.
"""

from pathlib import Path
import logging


PARSER_NAME = "gemini"
logger = logging.getLogger("tic_tac_go.daily_solve")


class DailyBoardParseError(RuntimeError):
    """Raised when the captured board image cannot be parsed."""


def parse_board_from_screenshot(screenshot_path: Path) -> list[list[str]]:
    """Parse a Tic Tac Go board from a screenshot and return it as a 2-D grid.

    Each cell in the returned grid is one of: "", "X", "O", "U", or "B".
    Raises DailyBoardParseError if the parser can't be loaded or the image
    can't be interpreted.
    """
    # The Gemini parser lives inside the solver package which depends on
    # google-generativeai. We import it here rather than at module top so the
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
