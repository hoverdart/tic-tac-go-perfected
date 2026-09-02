from types import SimpleNamespace
import unittest
from unittest.mock import patch

from solver.service import (
    CUSTOM_PUSH_MAX_NODES,
    CUSTOM_PUSH_TIMEOUT_SECONDS,
    SolverError,
    solve_custom_board,
)


class CustomSolverTests(unittest.TestCase):
    def test_solves_a_valid_rectangular_board_without_fallback(self):
        result = solve_custom_board([
            ["U", "", ""],
            ["O", "O", ""],
            ["", "", ""],
        ])
        self.assertTrue(result["solved"])
        self.assertEqual(result["solver_name"], "push-v3-custom")
        self.assertEqual(result["moves"], "RRD")

    def test_rejects_invalid_shape_before_search(self):
        with self.assertRaisesRegex(SolverError, "rectangular"):
            solve_custom_board([
                ["U", "", ""],
                ["O", "O", "", ""],
                ["", "", ""],
            ])
        with self.assertRaisesRegex(SolverError, "3x3"):
            solve_custom_board([["U", "O"]])

    def test_uses_fixed_public_search_caps_and_returns_timeout_result(self):
        timed_out = SimpleNamespace(moves=None, final_board=None, nodes_expanded=100_000)
        with patch("solver.service._run_push_solver", return_value=timed_out) as run:
            result = solve_custom_board([
                ["U", "", ""],
                ["O", "O", ""],
                ["", "", ""],
            ])
        self.assertFalse(result["solved"])
        self.assertEqual(result["states_checked"], 100_000)
        self.assertEqual(run.call_args.kwargs["max_nodes"], CUSTOM_PUSH_MAX_NODES)
        self.assertEqual(run.call_args.kwargs["timeout_seconds"], CUSTOM_PUSH_TIMEOUT_SECONDS)
        self.assertEqual(run.call_args.kwargs["quality_timeout_seconds"], 0.0)
