import unittest
from unittest.mock import patch

from solver import optimized_solver
from solver.gymnasium_register import ranked_real_boards
from solver.service import SolverError, solve_board


class SolverServiceTest(unittest.TestCase):
    def test_solved_board_returns_structured_result(self):
        result = solve_board(
            [
                ["U", "O", "O"],
                ["", "", ""],
                ["", "", ""],
            ]
        )

        self.assertTrue(result["solved"])
        self.assertEqual(result["moves"], "")
        self.assertEqual(result["states_checked"], 1)
        self.assertEqual(result["start_board"][0], ["U", "O", "O"])
        self.assertEqual(result["final_board"][0], ["U", "O", "O"])
        self.assertEqual(result["steps"], [])

    def test_invalid_board_raises_solver_error(self):
        with self.assertRaises(SolverError):
            solve_board(
                [
                    ["", "O", "O"],
                    ["", "", ""],
                    ["", "", ""],
                ]
            )

    def test_optimized_solver_solves_recent_daily_board(self):
        board = (
            ("", "X", "", "O", "", "B", "B", "B"),
            ("X", "", "X", "", "", "B", "B", "B"),
            ("", "O", "X", "", "X", "B", "B", "B"),
            ("U", "", "", "X", "", "B", "B", "B"),
            ("B", "B", "B", "B", "B", "B", "B", "B"),
            ("B", "B", "B", "B", "B", "B", "B", "B"),
            ("B", "B", "B", "B", "B", "B", "B", "B"),
            ("B", "B", "B", "B", "B", "B", "B", "B"),
        )

        moves, final_board, states_checked = optimized_solver.solve(
            board,
            progress_every=0,
            max_states=25_000,
        )

        self.assertIsNotNone(moves)
        self.assertIsNotNone(final_board)
        self.assertGreaterEqual(states_checked, 1)

    def test_optimized_solver_solves_ranked_regression_board(self):
        board = ranked_real_boards.fiveBoards[0]

        moves, final_board, states_checked = optimized_solver.solve(
            board,
            progress_every=0,
            max_states=10_000,
        )

        self.assertIsNotNone(moves)
        self.assertIsNotNone(final_board)
        self.assertEqual(len(moves), 6)
        self.assertGreaterEqual(states_checked, 1)

    def test_service_can_select_optimized_solver(self):
        with patch.dict("os.environ", {"SOLVER_IMPL": "optimized"}, clear=False):
            result = solve_board(
                [
                    ["U", "O", "O"],
                    ["", "", ""],
                    ["", "", ""],
                ]
            )

        self.assertTrue(result["solved"])
        self.assertEqual(result["moves"], "")
        self.assertEqual(result["states_checked"], 1)


if __name__ == "__main__":
    unittest.main()
