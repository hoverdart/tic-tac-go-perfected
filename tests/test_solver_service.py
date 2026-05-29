import unittest

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


if __name__ == "__main__":
    unittest.main()
