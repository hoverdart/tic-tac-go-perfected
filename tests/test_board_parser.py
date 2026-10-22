from datetime import date
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from apps.api.board_parser import resolve_daily_board, DailyBoardParseError
from apps.api.board_capture import extract_visible_puzzle_date
from apps.api.puzzle_catalog import CatalogError, parse_catalog
from solver.boardParsers import fallbackBoardParser as gemini

TARGET = date(2026, 10, 21)
BOARD = [["U", "O", "O"]]


class BoardParserTest(unittest.TestCase):
    def test_catalog_success_never_starts_browser_or_calls_gemini(self):
        snapshot = parse_catalog((Path(__file__).parent / "fixtures/catalog-sample.js").read_text(), source_url="https://www.gstatic.com/bundle")
        with (
            patch("apps.api.board_parser.entry_for_date", return_value=(snapshot.entries[TARGET], snapshot)),
            patch("apps.api.board_capture.capture_google_board_screenshot") as capture,
            patch("apps.api.board_parser.parse_board_from_screenshot") as vision,
        ):
            result = resolve_daily_board(TARGET)
        self.assertEqual(result.parser_name, "catalog")
        self.assertEqual(result.board[3][3], "B")
        self.assertEqual(result.source_url, snapshot.source_url)
        capture.assert_not_called()
        vision.assert_not_called()

    def test_catalog_failure_uses_date_matched_gemini_capture(self):
        with (
            patch("apps.api.board_parser.entry_for_date", side_effect=CatalogError("offline")),
            patch("apps.api.board_capture.capture_google_board_screenshot", return_value=SimpleNamespace(
                screenshot_path=Path("test.png"), puzzle_title="Captured", puzzle_date=TARGET)),
            patch("apps.api.board_parser.parse_board_from_screenshot", return_value=BOARD) as vision,
        ):
            result = resolve_daily_board(TARGET)
        self.assertEqual(result.parser_name, "gemini")
        self.assertEqual(result.board, BOARD)
        vision.assert_called_once_with(Path("test.png"))

    def test_unknown_or_wrong_capture_date_never_reaches_gemini(self):
        for captured in [None, date(2026, 10, 20)]:
            with (
                self.subTest(captured=captured),
                patch("apps.api.board_parser.entry_for_date", side_effect=CatalogError("missing")),
                patch("apps.api.board_capture.capture_google_board_screenshot", return_value=SimpleNamespace(
                    screenshot_path=Path("test.png"), puzzle_title=None, puzzle_date=captured)),
                patch("apps.api.board_parser.parse_board_from_screenshot") as vision,
                self.assertRaisesRegex(DailyBoardParseError, "does not match"),
            ):
                resolve_daily_board(TARGET)
            vision.assert_not_called()

    def test_date_extraction_rejects_ambiguous_or_invalid_dates(self):
        for text, expected in [("Tic Tac Go 10/21/2026 Test", TARGET),
                               ("10/21/2026 10/21/2026", TARGET),
                               ("10/20/2026 10/21/2026", None),
                               ("2/30/2026", None), ("No date", None)]:
            page = unittest.mock.Mock()
            page.locator.return_value.inner_text.return_value = text
            self.assertEqual(extract_visible_puzzle_date(page), expected)

    def test_gemini_respects_model_configuration_and_structural_retry(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image:
            with (
                patch.dict("os.environ", {"GEMINI_MODEL": "configured-model"}),
                patch.object(gemini, "_call_gemini", side_effect=["bad JSON", '{"board":[["U","O","O"],[".",".","."],[".",".","."]]}']) as call,
            ):
                result = gemini.parse_board_from_image(image.name)
        self.assertEqual(result[0][:3], ["U", "O", "O"])
        self.assertEqual(call.call_count, 2)
        self.assertTrue(all(c.args[2] == "configured-model" for c in call.call_args_list))


if __name__ == "__main__":
    unittest.main()
