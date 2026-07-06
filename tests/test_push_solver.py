import unittest

from solver.push_solver.core import (
    State,
    goal_info,
    is_deadlock,
    is_x_loss,
    normalize_state,
    parse_board,
    reachable,
    solve,
    successors,
)
from solver.push_solver.verify import verify_solution


class PushSolverTest(unittest.TestCase):
    def test_parse_board_builds_horizontal_and_vertical_lines_only(self):
        static, _state, _board, _player = parse_board(
            [
                ["U", "", ""],
                ["", "O", ""],
                ["", "", "O"],
            ]
        )

        coords = {
            tuple(static.coord(cell) for cell in line)
            for line in static.win_lines
        }

        self.assertIn(((0, 0), (0, 1), (0, 2)), coords)
        self.assertIn(((0, 0), (1, 0), (2, 0)), coords)
        self.assertNotIn(((0, 0), (1, 1), (2, 2)), coords)

    def test_reachable_and_normalize_merge_walk_equivalent_positions(self):
        static, state, _board, _player = parse_board(
            [
                ["U", "", "", "B"],
                ["", "B", "", ""],
                ["O", "", "X", "O"],
            ]
        )
        region = reachable(state.player, state.os, state.xs, static)
        alternate_player = max(region)

        normalized = normalize_state(alternate_player, state.os, state.xs, static)

        self.assertEqual(normalized, state)

    def test_successors_include_legal_o_and_x_pushes(self):
        static, state, _board, _player = parse_board(
            [
                ["", "", "", ""],
                ["U", "O", "", ""],
                ["", "X", "", ""],
                ["", "", "O", ""],
            ]
        )

        pushes = {(push.piece, push.cell, push.move) for push, _ in successors(state, static)}

        self.assertIn(("O", static.index(1, 1), "R"), pushes)
        self.assertIn(("X", static.index(2, 1), "R"), pushes)

    def test_successors_block_wall_and_piece_pushes(self):
        static, state, _board, _player = parse_board(
            [
                ["U", "O", "B", ""],
                ["", "X", "O", ""],
                ["", "", "", ""],
            ]
        )

        pushes = {(push.piece, push.cell, push.move) for push, _ in successors(state, static)}

        self.assertNotIn(("O", static.index(0, 1), "R"), pushes)
        self.assertNotIn(("X", static.index(1, 1), "R"), pushes)

    def test_goal_accepts_reachable_third_cell(self):
        static, state, _board, _player = parse_board(
            [
                ["U", "", ""],
                ["O", "O", ""],
                ["", "", ""],
            ]
        )

        self.assertIsNotNone(goal_info(state, static))

    def test_goal_rejects_blocked_third_cell(self):
        static, state, _board, _player = parse_board(
            [
                ["U", "B", ""],
                ["O", "O", "X"],
                ["", "", ""],
            ]
        )

        self.assertIsNone(goal_info(state, static))

    def test_x_loss_detects_three_x_line(self):
        static, _state, _board, _player = parse_board(
            [
                ["U", "", "O"],
                ["X", "X", "X"],
                ["", "", "O"],
            ]
        )

        self.assertTrue(
            is_x_loss(
                frozenset(
                    {
                        static.index(1, 0),
                        static.index(1, 1),
                        static.index(1, 2),
                    }
                ),
                static,
            )
        )

    def test_deadlock_prunes_static_dead_o_cell(self):
        static, _state, _board, _player = parse_board(
            [
                ["B", "B", "B", "B"],
                ["B", "U", "", "B"],
                ["B", "", "O", "B"],
                ["B", "B", "B", "B"],
            ]
        )

        self.assertTrue(is_deadlock(frozenset({static.index(2, 2)}), frozenset(), static))

    def test_solver_returns_verified_solution(self):
        board = [
            ["U", "", ""],
            ["O", "", ""],
            ["O", "", ""],
        ]

        result = solve(board, weight=1.0, max_nodes=1_000, timeout_seconds=1.0)
        verification = verify_solution(board, result.moves)

        self.assertTrue(result.solved)
        self.assertEqual(len(result.pushes), 0)
        self.assertTrue(verification.ok, verification.error)

    def test_solver_solves_one_push_board(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]

        result = solve(board, weight=1.0, max_nodes=1_000, timeout_seconds=1.0)
        verification = verify_solution(board, result.moves)

        self.assertTrue(result.solved)
        self.assertEqual(len(result.pushes), 1)
        self.assertTrue(verification.ok, verification.error)

    def test_solver_fails_cleanly_on_unsolvable_board(self):
        result = solve(
            [
                ["B", "B", "B", "B"],
                ["B", "U", "", "B"],
                ["B", "", "O", "B"],
                ["B", "B", "B", "B"],
            ],
            max_nodes=100,
            timeout_seconds=1.0,
        )

        self.assertFalse(result.solved)
        self.assertIn(result.failure_reason, {"deadlock", "exhausted"})


if __name__ == "__main__":
    unittest.main()
