from datetime import date
import unittest

from apps.api.title_fetcher import title_from_past_days


class TitleFetcherTest(unittest.TestCase):
    def test_title_from_past_days_accepts_date(self):
        self.assertEqual(title_from_past_days(date(2025, 1, 1)), "Tutorial")

    def test_title_from_past_days_accepts_iso_string(self):
        self.assertEqual(title_from_past_days("2025-10-07"), "Bendable")

    def test_title_from_past_days_returns_none_for_unknown_date(self):
        self.assertIsNone(title_from_past_days("1999-01-01"))


if __name__ == "__main__":
    unittest.main()
