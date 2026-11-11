"""The independent proof checker must find shorter paths before certifying."""

import random
import unittest

from scripts.check_optimized_paths_independently import check_path
from scripts.stress_optimized_paths import unit_bfs
from solver.push_solver.verify import verify_solution


class IndependentPathCheckerTest(unittest.TestCase):
    def test_simple_push_is_proved(self):
        result = check_path([["U", "O", "", "O"]], "R")
        self.assertEqual(result["status"], "confirmed")

    def test_walking_detour_is_rejected_as_shortest(self):
        board = [[""] * 5 for _ in range(3)]
        board[0][4], board[1][1], board[1][2] = "U", "O", "O"
        result = check_path(board, "LLLLD")
        self.assertEqual(result["status"], "shorter_found")
        self.assertEqual(len(result["shorter_path"]), 2)
        self.assertTrue(verify_solution(board, result["shorter_path"]).ok)
        self.assertEqual(
            check_path(board, result["shorter_path"])["status"], "confirmed"
        )

    def test_ooo_finish_and_zero_move_win(self):
        self.assertEqual(
            check_path([["U", "O", "", "O", "O"]], "R")["status"], "confirmed"
        )
        self.assertEqual(
            check_path([["U", "", "O", "O", "O"]], "")["status"], "confirmed"
        )

    def test_push_repair_finds_a_shorter_path(self):
        board = [
            ["X", "B", "U", ""],
            ["", "", "O", ""],
            ["", "O", "", ""],
            ["X", "", "", ""],
        ]
        result = check_path(board, "RDDLULLDRUR")
        self.assertEqual(result["status"], "shorter_found")
        self.assertLess(len(result["shorter_path"]), 11)
        self.assertTrue(verify_solution(board, result["shorter_path"]).ok)

    def test_interruption_is_inconclusive(self):
        board = [
            ["X", "B", "U", ""],
            ["", "", "O", ""],
            ["", "O", "", ""],
            ["X", "", "", ""],
        ]
        self.assertEqual(
            check_path(board, "RDDLULLDRUR", seconds=0)["status"],
            "inconclusive_timeout",
        )
        self.assertEqual(
            check_path(board, "RDDLULLDRUR", max_nodes=0)["status"],
            "inconclusive_node_cap",
        )

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            check_path([["U", "O", "", "O"]], "U")

    def test_generated_longer_paths_cannot_receive_false_confirmation(self):
        randomizer = random.Random(919)
        checked = nonoptimal = 0
        for index in range(40):
            board = [[""] * 4 for _ in range(3)]
            pieces = ["U", "O", "O", "X", "X", "B"]
            if index % 3 == 0:
                pieces.append("O")
            for cell, piece in zip(randomizer.sample(range(12), len(pieces)), pieces):
                board[cell // 4][cell % 4] = piece
            oracle = unit_bfs(board, max_nodes=10_000, goals=8)
            if not oracle["answers"]:
                continue
            optimum = len(oracle["answers"][0])
            candidate = max(oracle["answers"], key=len)
            result = check_path(board, candidate, seconds=0.1, max_nodes=10_000)
            if result["status"] == "confirmed":
                self.assertEqual(len(candidate), optimum)
            elif result["status"] == "shorter_found":
                self.assertGreaterEqual(len(result["shorter_path"]), optimum)
            checked += 1
            nonoptimal += len(candidate) > optimum
        self.assertGreaterEqual(checked, 10)
        self.assertGreaterEqual(nonoptimal, 5)
