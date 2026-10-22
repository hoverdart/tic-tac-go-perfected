from datetime import date
from pathlib import Path
import unittest

from apps.api.puzzle_catalog import parse_catalog
from scripts.audit_catalog import audit, canonical_board


class CatalogAuditTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = parse_catalog((Path(__file__).parent / "fixtures/catalog-sample.js").read_text())

    def test_known_false_success_is_excluded_from_training(self):
        board = canonical_board(self.snapshot.entries[date(2026, 10, 21)].board())
        board[2][2] = ""
        board = [row + ["B"] * 2 for row in board] + [["B"] * 8] * 2
        records = [{"puzzle_date": "2026-10-21", "status": "solved", "board": board, "moves": "RULUL"}]
        _, rows, positives, summary = audit(self.snapshot, records, date(2026, 10, 21))
        self.assertTrue(rows[0]["valid_on_stored"])
        self.assertFalse(rows[0]["valid_on_official"])
        self.assertTrue(rows[0]["mismatch"])
        self.assertEqual(positives, [])
        self.assertEqual(summary["false_solved"], 1)

    def test_future_boards_and_paths_never_enter_exports(self):
        records = [{"puzzle_date": "2026-10-21", "status": "solved", "board": None, "moves": "RULUL"}]
        manifest, rows, positives, _ = audit(self.snapshot, records, date(2026, 10, 20))
        self.assertTrue(all(row["id"] <= "20261020" for row in manifest))
        self.assertEqual(rows, [])
        self.assertEqual(positives, [])

    def test_outer_border_translation_preserves_playable_layout(self):
        official = self.snapshot.entries[date(2026, 10, 21)].board()
        visible = canonical_board(official)
        shifted = [row + ["B"] * 2 for row in visible] + [["B"] * 8] * 2
        self.assertEqual(canonical_board(official), canonical_board(shifted))
        self.assertEqual(canonical_board(official)[2][2], "B")


if __name__ == "__main__":
    unittest.main()
