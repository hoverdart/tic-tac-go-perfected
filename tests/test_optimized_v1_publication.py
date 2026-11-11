"""Public paths cannot regress, misidentify boards, or overstate proof status."""

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import date
from pathlib import Path

from apps.api.puzzle_catalog import CatalogEntry
from scripts.update_optimized_v1 import (
    choose_record,
    first_win,
    make_record,
    read_feed,
    save_feed,
    validate_record,
)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.entry = CatalogEntry(date(2026, 11, 11), "Fixture", 4, 3, "P-----OO----")
        self.short = make_record(self.entry, "RD", proof=True, lower_bound=2)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.json"
            save_feed(path, {"2026-11-11": self.short})
            self.assertEqual(read_feed(path), {"2026-11-11": self.short})
            value = json.loads(path.read_text())
            value["solutions"].append(deepcopy(self.short))
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_feed(path)

    def test_longer_paths_do_not_replace_shorter(self):
        long = make_record(self.entry, "DDRU")
        self.assertIs(choose_record(self.short, long), self.short)
        self.assertIs(choose_record(long, self.short), self.short)

    def test_equal_length_certificate_upgrades_path(self):
        unproved = make_record(self.entry, "RD")
        self.assertIs(choose_record(unproved, self.short), self.short)
        self.assertIs(choose_record(self.short, unproved), self.short)

    def test_contradictory_certificate_fails(self):
        long = make_record(self.entry, "DDRU", proof=True, lower_bound=4)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            choose_record(self.short, long)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            choose_record(long, self.short)

    def test_replay_rejects_blocked_or_unsolved_moves(self):
        for moves in ("L", "U", "", "bad"):
            with self.subTest(moves=moves), self.assertRaises(ValueError):
                make_record(self.entry, moves)

    def test_first_win_removes_post_win_moves(self):
        self.assertEqual(first_win(self.entry.board(), "RDLR"), "RD")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            make_record(self.entry, "RDLR", proof=True, lower_bound=4)

    def test_invalid_bound_and_geometry_are_rejected(self):
        invalid = deepcopy(self.short)
        invalid["lower_bound"] = 3
        with self.assertRaises(ValueError):
            validate_record(invalid)
        invalid = deepcopy(self.short)
        invalid["board"][7][7] = ""
        with self.assertRaisesRegex(ValueError, "geometry"):
            choose_record(self.short, invalid)

    def test_initial_winning_board_uses_zero_moves(self):
        entry = CatalogEntry(date(2026, 11, 11), "Won", 3, 3, "POO------")
        self.assertEqual(make_record(entry, "", proof=True)["moves"], "")


if __name__ == "__main__":
    unittest.main()
