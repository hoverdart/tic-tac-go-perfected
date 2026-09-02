import unittest
from unittest.mock import patch

from fastapi import HTTPException

from apps.api.main import SolveRequest, custom_solve


VALID_BOARD = [["U", "", ""], ["O", "O", ""], ["", "", ""]]
VALID_HASH = "a" * 64
RESULT = {
    "solved": True,
    "solver_name": "push-v3-custom",
    "moves": "RRD",
    "states_checked": 3,
    "elapsed_ms": 2.0,
    "start_board": VALID_BOARD,
    "final_board": [["", "", ""], ["O", "O", "U"], ["", "", ""]],
    "steps": [],
}


class CustomSolveEndpointTests(unittest.TestCase):
    def test_cache_hit_skips_allowance_and_solver(self):
        with (
            patch("apps.api.main.get_cached_custom_solution", return_value=RESULT),
            patch("apps.api.main.reserve_custom_solve") as reserve,
            patch("apps.api.main.solve_custom_board") as solve,
        ):
            response = custom_solve(SolveRequest(board=VALID_BOARD), VALID_HASH)

        self.assertTrue(response.cached)
        self.assertIsNone(response.remaining)
        reserve.assert_not_called()
        solve.assert_not_called()

    def test_new_solve_consumes_one_allowance_then_caches_verified_result(self):
        with (
            patch("apps.api.main.get_cached_custom_solution", return_value=None),
            patch("apps.api.main.reserve_custom_solve", return_value=1),
            patch("apps.api.main.solve_custom_board", return_value=RESULT),
            patch("apps.api.main.cache_custom_solution") as cache,
        ):
            response = custom_solve(SolveRequest(board=VALID_BOARD), VALID_HASH)

        self.assertFalse(response.cached)
        self.assertEqual(response.remaining, 9)
        cache.assert_called_once()

    def test_exhausted_allowance_returns_429_and_retry_after(self):
        with patch("apps.api.main.get_cached_custom_solution", return_value=None), patch(
            "apps.api.main.reserve_custom_solve", return_value=None
        ), self.assertRaises(HTTPException) as raised:
            custom_solve(SolveRequest(board=VALID_BOARD), VALID_HASH)

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers, {"Retry-After": "3600"})

    def test_requires_hmac_derived_identity_shape(self):
        with self.assertRaisesRegex(HTTPException, "invalid custom solver client identity"):
            custom_solve(SolveRequest(board=VALID_BOARD), "raw-ip-address")
