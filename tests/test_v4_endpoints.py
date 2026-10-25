from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from apps.api.main import app, _benchmark_lock
from apps.api.puzzle_catalog import parse_catalog


class V4EndpointsTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.auth = {"Authorization": "Bearer test-secret"}
        self.env = patch.dict("os.environ", {"CRON_SECRET": "test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.snapshot = parse_catalog(
            (Path(__file__).parent / "fixtures/catalog-sample.js").read_text()
        )

    def test_benchmark_auth_limits_and_no_persistence_or_fallback(self):
        from solver.push_solver import solve_v4

        result = solve_v4([["U", "O", "O"]], quality_timeout_seconds=0)
        with (
            patch(
                "apps.api.main.entry_for_date",
                return_value=(self.snapshot.entries[date(2026, 10, 21)], self.snapshot),
            ) as catalog,
            patch("solver.push_solver.solve_v4", return_value=result) as solve,
            patch(
                "solver.push_solver.verify_solution",
                return_value=type("Check", (), {"ok": True})(),
            ),
            patch("apps.api.main.run_daily_solve") as daily,
            patch("solver.service._run_heuristic_cnn_solver") as fallback,
        ):
            self.assertEqual(
                self.client.post(
                    "/debug/push-benchmark/2026-10-21", json={}
                ).status_code,
                401,
            )
            catalog.assert_not_called()
            self.assertEqual(
                self.client.post(
                    "/debug/push-benchmark/2026-10-21",
                    json={"timeout_seconds": 31},
                    headers=self.auth,
                ).status_code,
                422,
            )
            response = self.client.post(
                "/debug/push-benchmark/2026-10-21", json={}, headers=self.auth
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["verified"])
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(solve.call_args.kwargs["quality_timeout_seconds"], 0)
            daily.assert_not_called()
            fallback.assert_not_called()

    def test_concurrent_benchmark_rejected_and_ledger_read_requires_auth(self):
        _benchmark_lock.acquire()
        try:
            self.assertEqual(
                self.client.post(
                    "/debug/push-benchmark/2026-10-21", json={}, headers=self.auth
                ).status_code,
                409,
            )
        finally:
            _benchmark_lock.release()
        with patch(
            "apps.api.main.list_daily_solve_runs", return_value=[{"status": "unsolved"}]
        ) as runs:
            self.assertEqual(
                self.client.get("/debug/solve-runs/2026-10-21").status_code, 401
            )
            runs.assert_not_called()
            self.assertEqual(
                self.client.get(
                    "/debug/solve-runs/2026-10-21", headers=self.auth
                ).json()["runs"],
                [{"status": "unsolved"}],
            )
