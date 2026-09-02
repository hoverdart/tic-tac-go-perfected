from datetime import date
import unittest

from apps.api.puzzle_titles import (
    _clean_title,
    clean_puzzle_title,
    normalize_official_title,
    official_level_label,
    title_from_official_catalog,
    title_from_past_days,
)
from unittest.mock import patch


class TitleFetcherTest(unittest.TestCase):
    def test_title_from_past_days_accepts_date(self):
        self.assertEqual(title_from_past_days(date(2025, 1, 1)), "Tutorial")

    def test_title_from_past_days_accepts_iso_string(self):
        self.assertEqual(title_from_past_days("2025-10-07"), "Bendable")

    def test_title_from_past_days_returns_none_for_unknown_date(self):
        self.assertIsNone(title_from_past_days("1999-01-01"))

    def test_trusted_manifest_preserves_short_and_punctuation_titles(self):
        self.assertEqual(title_from_past_days("2025-11-10"), "So Close")
        self.assertEqual(title_from_past_days("2025-12-12"), "-_-")
        self.assertEqual(normalize_official_title(" -_- "), "-_-" )

    def test_official_catalog_and_level_fallbacks_are_date_keyed(self):
        with patch("apps.api.puzzle_titles._official_catalog_titles", return_value={"20260901": "Level 2026-09-01"}):
            self.assertEqual(title_from_official_catalog("2026-09-01"), "Level 2026-09-01")
        self.assertEqual(official_level_label("2026-09-02"), "Level 2026-09-02")

    def test_clean_title_strips_google_search_title_noise(self):
        self.assertEqual(_clean_title("Tic Tac Go - Bendable - Google Search"), "Bendable")

    def test_clean_title_strips_plain_game_prefix(self):
        self.assertEqual(_clean_title("Tic Tac Go Bendable"), "Bendable")

    def test_clean_title_rejects_generic_game_title(self):
        self.assertIsNone(_clean_title("Tic Tac Go"))

    def test_clean_title_rejects_normalized_google_game_heading(self):
        self.assertIsNone(_clean_title("  A GOOGLE---GAME  "))
        self.assertIsNone(clean_puzzle_title("a Google game"))

    def test_clean_title_rejects_slash_dates(self):
        self.assertIsNone(_clean_title("5/14/2026"))
        self.assertIsNone(_clean_title("5 / 14 / 26"))

    def test_clean_title_keeps_real_title_with_numbers(self):
        self.assertEqual(_clean_title("Route 66"), "Route 66")

    def test_clean_title_does_not_strip_a_real_title_ending_in_close(self):
        self.assertEqual(_clean_title("So Close"), "So Close")

    def test_clean_title_extracts_title_from_combined_google_heading(self):
        heading = "Tic-Tac-Go a Google game 7/2/2026 Equator Undo Reset Rules"
        self.assertEqual(_clean_title(heading), "Equator")


if __name__ == "__main__":
    unittest.main()
