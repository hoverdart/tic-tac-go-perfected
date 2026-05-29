from pathlib import Path


PARSER_NAME = "gemini"


class DailyBoardParseError(RuntimeError):
    """Raised when the captured board image cannot be parsed."""


def parse_board_from_screenshot(screenshot_path: Path) -> list[list[str]]:
    try:
        from solver.boardParsers.fallbackBoardParser import parse_board_from_image
    except ImportError as exc:
        raise DailyBoardParseError("Could not load the Gemini board parser.") from exc

    try:
        return parse_board_from_image(screenshot_path)
    except Exception as exc:
        raise DailyBoardParseError(f"Could not parse Tic Tac Go board: {exc}") from exc
