from datetime import date
import unittest
from unittest.mock import patch, ANY

from apps.api.board_parser import DailyBoardInput, DailyBoardParseError
from apps.api.daily_solve import fallback_puzzle_title, run_daily_solve

TARGET = date(2026, 10, 21)
BOARD = [["U", "O", "O"]]
RESULT = {
    "solved": True, "solver_name": "push-v3", "moves": "", "states_checked": 1,
    "elapsed_ms": 2.5, "start_board": BOARD, "final_board": BOARD, "steps": [],
}


class DailySolveTest(unittest.TestCase):
    def setUp(self):
        begin = patch("apps.api.daily_solve.begin_daily_solve_run", return_value=42)
        finish = patch("apps.api.daily_solve.finish_daily_solve_run")
        begin.start()
        self.finish = finish.start()
        self.addCleanup(begin.stop)
        self.addCleanup(finish.stop)
        progress=patch("apps.api.daily_solve.update_daily_solve_run")
        self.progress=progress.start()
        self.addCleanup(progress.stop)

    def test_fallback_title_uses_official_level_label_when_catalog_is_unavailable(self):
        with (
            patch(
                "apps.api.daily_solve.title_from_official_catalog", return_value=None
            ),
            patch("apps.api.daily_solve.title_from_past_days", return_value=None),
        ):
            self.assertEqual(
                fallback_puzzle_title(date(2026, 9, 2)), "Level 2026-09-02"
            )

    def test_cron_pipeline_preserves_actual_parser_and_source(self):
        for parser_name in ["catalog", "gemini"]:
            with (
                self.subTest(parser=parser_name),
                patch(
                    "apps.api.daily_solve.resolve_daily_board",
                    return_value=DailyBoardInput(
                        BOARD,
                        "Test Puzzle",
                        parser_name,
                        "https://www.gstatic.com/versioned-bundle",
                    ),
                ) as resolve,
                patch(
                    "apps.api.daily_solve.solve_daily_board", return_value=RESULT
                ) as solve,
                patch(
                    "apps.api.daily_solve.upsert_solution",
                    side_effect=lambda r, **kw: r,
                ) as persist,
            ):
                record = run_daily_solve(TARGET)
            resolve.assert_called_once_with(TARGET)
            solve.assert_called_once_with(BOARD,on_push_complete=ANY)
            self.assertEqual(record["status"], "solved")
            self.assertEqual(record["parser_name"], parser_name)
            self.assertEqual(
                record["source_url"], "https://www.gstatic.com/versioned-bundle"
            )
            self.assertEqual(record["puzzle_date"], TARGET)
            persist.assert_called_once_with(record, preserve_solved=True)

    def test_unsolved_catalog_board_is_not_reparsed_with_gemini(self):
        with (
            patch(
                "apps.api.daily_solve.resolve_daily_board",
                return_value=DailyBoardInput(
                    BOARD, "Test", "catalog", "https://www.gstatic.com/bundle"
                ),
            ) as resolve,
            patch(
                "apps.api.daily_solve.solve_daily_board",
                return_value={**RESULT, "solved": False, "moves": None},
            ),
            patch(
                "apps.api.daily_solve.upsert_solution", side_effect=lambda r, **kw: r
            ),
        ):
            record = run_daily_solve(TARGET)
        self.assertEqual(record["status"], "unsolved")
        self.assertEqual(self.finish.call_args.kwargs["status"], "unsolved")
        self.assertEqual(record["parser_name"], "catalog")
        resolve.assert_called_once()

    def test_parse_failure_does_not_solve_and_requests_guarded_storage(self):
        with (
            patch(
                "apps.api.daily_solve.resolve_daily_board",
                side_effect=DailyBoardParseError("wrong date"),
            ),
            patch("apps.api.daily_solve.fallback_puzzle_title", return_value="Test"),
            patch("apps.api.daily_solve.solve_daily_board") as solve,
            patch(
                "apps.api.daily_solve.upsert_solution", side_effect=lambda r, **kw: r
            ) as persist,
        ):
            record = run_daily_solve(TARGET)
        solve.assert_not_called()
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["board"])
        persist.assert_called_once_with(record, preserve_solved=True)

    def test_unsuccessful_retry_keeps_its_receipt_when_solution_is_protected(self):
        with (
            patch(
                "apps.api.daily_solve.resolve_daily_board",
                return_value=DailyBoardInput(
                    BOARD, "Test", "catalog", "https://example.test/catalog", "abc123"
                ),
            ),
            patch(
                "apps.api.daily_solve.solve_daily_board",
                return_value={
                    **RESULT,
                    "solved": False,
                    "moves": None,
                    "diagnostics": {
                        "push_expanded_nodes": 17,
                        "fallback_reported_transitions": 50,
                    },
                },
            ),
            patch(
                "apps.api.daily_solve.upsert_solution",
                return_value={"status": "solved"},
            ),
        ):
            record = run_daily_solve(TARGET)
        self.assertEqual(record["status"], "solved")
        receipt = self.finish.call_args.kwargs
        self.assertEqual(receipt["status"], "unsolved")
        self.assertEqual(receipt["board_sha256"], "abc123")
        self.assertEqual(receipt["diagnostics"]["push_expanded_nodes"], 17)
        self.assertEqual(receipt["diagnostics"]["stored_status"], "solved")


if __name__ == "__main__":
    unittest.main()
