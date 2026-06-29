from datetime import date
import unittest

from apps.api.puzzle_titles import _clean_title, title_from_past_days


class TitleFetcherTest(unittest.TestCase):
    def test_title_from_past_days_accepts_date(self):
        self.assertEqual(title_from_past_days(date(2025, 1, 1)), "Tutorial")

    def test_title_from_past_days_accepts_iso_string(self):
        self.assertEqual(title_from_past_days("2025-10-07"), "Bendable")

    def test_title_from_past_days_returns_none_for_unknown_date(self):
        self.assertIsNone(title_from_past_days("1999-01-01"))

    def test_clean_title_strips_google_search_title_noise(self):
        self.assertEqual(_clean_title("Tic Tac Go - Bendable - Google Search"), "Bendable")

    def test_clean_title_strips_plain_game_prefix(self):
        self.assertEqual(_clean_title("Tic Tac Go Bendable"), "Bendable")

    def test_clean_title_rejects_generic_game_title(self):
        self.assertIsNone(_clean_title("Tic Tac Go"))


if __name__ == "__main__":
    unittest.main()
