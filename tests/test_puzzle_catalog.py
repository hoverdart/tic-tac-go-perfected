from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

from apps.api import puzzle_catalog as catalog

FIXTURE = Path(__file__).parent / "fixtures" / "catalog-sample.js"
SAMPLE = '["20261021",{name:"Test",I:new Z(2026,10,21),width:3,height:3,level:"POO------"}]'


class PuzzleCatalogTest(unittest.TestCase):
    def test_real_entries_preserve_walls_dates_and_multiple_os(self):
        snapshot = catalog.parse_catalog(FIXTURE.read_text())
        self.assertEqual(len(snapshot.entries), 9)
        entry = snapshot.entries[date(2026, 10, 21)]
        self.assertEqual(entry.title, "Too Close For Comfort")
        self.assertEqual(entry.board()[3][3], "B")
        self.assertEqual(entry.board()[4][4], "U")
        self.assertEqual(snapshot.entries[date(2025, 10, 23)].level.count("O"), 6)
        self.assertEqual(len(snapshot.source_sha256), 64)
        self.assertEqual(len(entry.board_sha256), 64)
        # Returning mutable wire data cannot mutate the cached source.
        entry.board()[3][3] = ""
        self.assertEqual(entry.board()[3][3], "B")

    def test_rectangular_input_whitespace_and_escaped_names(self):
        source = SAMPLE.replace('"Test"', r'"A \"quote\""').replace(
            'width:3,height:3,level:"POO------"',
            'width:4,\n height:3,\n level:"POO---------"',
        ).replace('21),', '21),\n')
        entry = catalog.parse_catalog(source).entries[date(2026, 10, 21)]
        self.assertEqual(entry.title, 'A "quote"')
        self.assertEqual(entry.board()[0], ["U", "O", "O", "", "B", "B", "B", "B"])
        self.assertEqual(entry.board()[3], ["B"] * 8)

    def test_rejects_bad_or_partial_catalog_instead_of_guessing(self):
        bad = ["", SAMPLE + SAMPLE, SAMPLE.replace("2026,10,21", "2026,10,20"),
               SAMPLE.replace("2026,10,21", "2026,2,30"),
               SAMPLE.replace("width:3", "width:9"),
               SAMPLE.replace("POO------", "POO-----"),
               SAMPLE.replace("POO------", "POO----Z-"),
               SAMPLE.replace("POO------", "PPO------"),
               SAMPLE.replace("POO------", "PO-------"),
               SAMPLE.replace('name:"Test"', 'name:""'),
               SAMPLE + '["20261022",{name:"Unrecognized",changed:true}]']
        for source in bad:
            with self.subTest(source=source), self.assertRaises(catalog.CatalogError):
                catalog.parse_catalog(source)

    def test_loader_discovers_versioned_bundle_and_rejects_other_hosts(self):
        html = '<script src="https://www.gstatic.com/GoogleGamesPlatformUi/m=_b"></script>'
        with patch.object(catalog, "_fetch_text", side_effect=[html, SAMPLE]) as fetch:
            result = catalog.fetch_catalog()
        self.assertTrue(result.source_url.endswith("m=Aeygpd"))
        self.assertEqual(fetch.call_count, 2)
        with patch.object(catalog, "_fetch_text", return_value=html.replace("www.gstatic.com", "example.test")):
            with self.assertRaises(catalog.CatalogError):
                catalog.fetch_catalog()

    def test_network_failures_have_consistent_error_type(self):
        with patch.object(catalog, "_fetch_text", side_effect=TimeoutError("offline")):
            with self.assertRaisesRegex(catalog.CatalogError, "offline"):
                catalog.fetch_catalog()

    def test_cache_refreshes_success_and_recovers_after_failure_backoff(self):
        snapshot = catalog.parse_catalog(SAMPLE)
        with (
            patch.object(catalog, "_snapshot", None),
            patch.object(catalog, "_expires_at", 0),
            patch.object(catalog, "_retry_after", 0),
            patch.object(catalog.time, "monotonic", return_value=100) as clock,
            patch.object(catalog, "fetch_catalog", side_effect=[catalog.CatalogError("offline"), snapshot, snapshot]) as fetch,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(catalog.CatalogError, "offline"):
                    catalog.load_catalog()
            self.assertEqual(fetch.call_count, 1)
            clock.return_value = 131
            self.assertIs(catalog.load_catalog(), snapshot)
            self.assertIs(catalog.load_catalog(), snapshot)
            self.assertEqual(fetch.call_count, 2)
            clock.return_value = 3732
            self.assertIs(catalog.load_catalog(), snapshot)
            self.assertEqual(fetch.call_count, 3)

    def test_missing_date_is_not_replaced_with_a_neighbor(self):
        with patch.object(catalog, "load_catalog", return_value=catalog.parse_catalog(SAMPLE)):
            with self.assertRaisesRegex(catalog.CatalogError, "2026-10-20"):
                catalog.entry_for_date(date(2026, 10, 20))


if __name__ == "__main__":
    unittest.main()
