from datetime import date
import unittest

from backfill_solutions import ALL_PAST_DAYS, board_from_entry, build_record, parse_entry_date


class BackfillSolutionsTest(unittest.TestCase):
    def test_parse_entry_date_prefers_manifest_date(self):
        self.assertEqual(parse_entry_date(ALL_PAST_DAYS[0]), date(2025, 1, 1))

    def test_board_from_entry_maps_manifest_cells(self):
        board = board_from_entry(ALL_PAST_DAYS[0])

        self.assertEqual(len(board), 8)
        self.assertEqual(len(board[0]), 8)
        self.assertEqual(board[3][6], "O")
        self.assertEqual(board[4][2], "U")
        self.assertEqual(board[0][0], "B")
        self.assertEqual(board[3][3], "")

    def test_build_record_for_timeout_stores_failed_row_shape(self):
        board = board_from_entry(ALL_PAST_DAYS[0])
        record = build_record(
            ALL_PAST_DAYS[0],
            date(2025, 1, 1),
            board,
            {
                "ok": False,
                "timed_out": True,
                "error_message": "Optimized solver exceeded 60.0 seconds.",
                "elapsed_ms": 60_000,
            },
            "hybrid",
        )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["parser_name"], "backfill_solutions")
        self.assertEqual(record["solver_name"], "optimized-hybrid")
        self.assertEqual(record["puzzle_title"], "Tutorial")
        self.assertEqual(record["board"], board)


if __name__ == "__main__":
    unittest.main()
