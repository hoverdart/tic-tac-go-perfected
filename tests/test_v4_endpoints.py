import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api.main import _benchmark_lock, app
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

    def test_fresh_worker_endpoint_is_read_only_and_reports_startup_time(self):
        with (
            patch("apps.api.main.entry_for_date", return_value=(self.snapshot.entries[date(2026, 10, 21)], self.snapshot)),
            patch("apps.api.main.run_daily_solve") as daily,
            patch("solver.service._run_heuristic_cnn_solver") as fallback,
        ):
            response = self.client.post(
                "/debug/push-benchmark/2026-10-21",
                json={"process": "fresh", "schedule": "coverage", "timeout_seconds": 5, "max_nodes": 1000},
                headers=self.auth,
            )
        self.assertEqual(response.status_code, 200, response.text)
        receipt = response.json()
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["process"], "fresh")
        self.assertGreater(receipt["process_wall_seconds"], receipt["wall_seconds"])
        self.assertEqual(receipt["limits"], {"seconds": 5, "nodes": 1000})
        daily.assert_not_called()
        fallback.assert_not_called()

    def test_runtime_config_is_authenticated_and_uses_the_daily_runner_settings(self):
        self.assertEqual(self.client.get("/debug/solver-config").status_code, 401)
        with patch.dict("os.environ", {"DAILY_PUSH_SOLVER_VERSION": "v4"}):
            response = self.client.get("/debug/solver-config", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        daily = response.json()["daily"]
        self.assertEqual(daily, {"version": "v4", "schedule": "coverage", "timeout_seconds": 30, "quality_timeout_seconds": 30, "max_nodes": 500000})

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
            response = self.client.post(
                "/debug/push-benchmark/2026-10-21",
                json={
                    "schedule": "beam8",
                    "beam_mode": "serial",
                    "macro_mode": "forced",
                },
                headers=self.auth,
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(solve.call_args.kwargs["schedule"], "beam8")
            self.assertEqual(solve.call_args.kwargs["beam_mode"], "serial")
            self.assertEqual(solve.call_args.kwargs["macro_mode"], "forced")
            response = self.client.post(
                "/debug/push-benchmark/2026-10-21",
                json={"schedule": "coverage"},
                headers=self.auth,
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(solve.call_args.kwargs["schedule"], "coverage")
            response = self.client.post(
                "/debug/push-benchmark/2026-10-21",
                json={"schedule": "coverage", "quality_timeout_seconds": 10},
                headers=self.auth,
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(solve.call_args.kwargs["quality_timeout_seconds"], 10)
            for invalid in (
                {"schedule": "phs_beam"},
                {"policy": "phs-v1"},
                {"policy": "../../private"},
                {"version": "v3", "strategy": "phs_star"},
            ):
                self.assertEqual(
                    self.client.post(
                        "/debug/push-benchmark/2026-10-21",
                        json=invalid,
                        headers=self.auth,
                    ).status_code,
                    422,
                )
            response = self.client.post(
                "/debug/push-benchmark/2026-10-21",
                json={"strategy": "phs_star", "policy": "phs-mlp-v1"},
                headers=self.auth,
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(solve.call_args.kwargs["strategy"], "phs_star")
            self.assertEqual(
                solve.call_args.kwargs["policy_path"].name, "policy-mlp.json"
            )
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
