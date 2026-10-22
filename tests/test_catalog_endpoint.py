from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from apps.api.main import app
from apps.api.puzzle_catalog import CatalogError, parse_catalog


class CatalogEndpointTest(unittest.TestCase):
    def test_read_only_debug_endpoint_requires_auth_and_never_solves(self):
        snapshot = parse_catalog((Path(__file__).parent / "fixtures/catalog-sample.js").read_text(), source_url="https://www.gstatic.com/bundle")
        with (
            patch.dict("os.environ", {"CRON_SECRET": "test-secret"}),
            patch("apps.api.main.entry_for_date", return_value=(snapshot.entries[date(2026, 10, 21)], snapshot)) as entry,
            patch("apps.api.main.run_daily_solve") as daily,
        ):
            client = TestClient(app)
            self.assertEqual(client.get("/debug/catalog/2026-10-21").status_code, 401)
            entry.assert_not_called()
            response = client.get("/debug/catalog/2026-10-21", headers={"Authorization": "Bearer test-secret"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["board"][3][3], "B")
            self.assertEqual(response.json()["source_url"], snapshot.source_url)
            self.assertEqual(response.headers["cache-control"], "no-store")
            daily.assert_not_called()

    def test_catalog_outage_is_visible_without_triggering_gemini(self):
        with (
            patch.dict("os.environ", {"CRON_SECRET": "test-secret"}),
            patch("apps.api.main.entry_for_date", side_effect=CatalogError("offline")),
            patch("apps.api.board_parser.parse_board_from_screenshot") as gemini,
        ):
            response = TestClient(app).get("/debug/catalog/2026-10-21", headers={"Authorization": "Bearer test-secret"})
            self.assertEqual(response.status_code, 503)
            gemini.assert_not_called()


if __name__ == "__main__":
    unittest.main()
