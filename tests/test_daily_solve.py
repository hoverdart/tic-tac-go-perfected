from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from apps.api.daily_solve import fallback_puzzle_title, run_daily_solve


class DailySolveTest(unittest.TestCase):
    def test_fallback_title_uses_official_level_label_when_catalog_is_unavailable(self):
        with (
            patch("apps.api.daily_solve.title_from_official_catalog", return_value=None),
            patch("apps.api.daily_solve.title_from_past_days", return_value=None),
        ):
            self.assertEqual(fallback_puzzle_title(date(2026, 9, 2)), "Level 2026-09-02")

    def test_cron_pipeline_uses_daily_push_beam_portfolio(self):
        board = [["U", "O", "O"]]
        solve_result = {
            "solved": True,
            "solver_name": "push-v3",
            "moves": "",
            "states_checked": 1,
            "elapsed_ms": 2.5,
            "start_board": board,
            "final_board": board,
            "steps": [],
        }

        with (
            patch(
                "apps.api.daily_solve.google_tic_tac_go_url",
                return_value="https://example.test/puzzle",
            ),
            patch(
                "apps.api.daily_solve.capture_google_board_screenshot",
                return_value=SimpleNamespace(
                    screenshot_path="/tmp/daily.png",
                    puzzle_title="Test Puzzle",
                ),
            ),
            patch(
                "apps.api.daily_solve.parse_board_from_screenshot",
                return_value=board,
            ),
            patch(
                "apps.api.daily_solve.solve_daily_board",
                return_value=solve_result,
            ) as solve_daily,
            patch(
                "apps.api.daily_solve.upsert_solution",
                side_effect=lambda record: record,
            ),
        ):
            record = run_daily_solve(date(2026, 7, 15))

        solve_daily.assert_called_once_with(board)
        self.assertEqual(record["status"], "solved")
        self.assertEqual(record["solver_name"], "push-v3")
        self.assertEqual(record["puzzle_title"], "Test Puzzle")


if __name__ == "__main__":
    unittest.main()
