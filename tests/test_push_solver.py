import math
import unittest
from unittest.mock import patch

from solver.push_solver import core
from solver.push_solver.core import (
    SearchStrategyConfig,
    State,
    goal_info,
    heuristic,
    is_deadlock,
    is_x_loss,
    normalize_state,
    parse_board,
    reachable,
    solve,
    solve_v1,
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

        pushes = {(push.piece, push.cell, push.move) for push, *_ in successors(state, static)}

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

        pushes = {(push.piece, push.cell, push.move) for push, *_ in successors(state, static)}

        self.assertNotIn(("O", static.index(0, 1), "R"), pushes)
        self.assertNotIn(("X", static.index(1, 1), "R"), pushes)

    def test_successors_returns_matching_heuristic_value(self):
        static, state, _board, _player = parse_board(
            [
                ["", "", "", ""],
                ["U", "O", "", ""],
                ["", "X", "", ""],
                ["", "", "O", ""],
            ]
        )

        for push, nxt, _region, h, _bias in successors(state, static):
            self.assertEqual(h, heuristic(nxt, static))

    def test_successors_reuses_provided_region(self):
        static, state, _board, _player = parse_board(
            [
                ["", "", "", ""],
                ["U", "O", "", ""],
                ["", "", "O", ""],
            ]
        )
        region = reachable(state.player, state.os, state.xs, static)

        call_count = []
        original_reachable = core.reachable

        def counting_reachable(*args, **kwargs):
            call_count.append(1)
            return original_reachable(*args, **kwargs)

        with patch.object(core, "reachable", side_effect=counting_reachable):
            results = core.successors(state, static, region=region)

        # normalize_state() calls reachable() once per generated successor;
        # passing region= must not add any call beyond that.
        self.assertEqual(len(call_count), len(results))

    def test_solve_does_not_recompute_heuristic_for_successors(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]

        heuristic_call_count = []
        successor_item_count = []
        original_heuristic = core.heuristic
        original_successors = core.successors

        def counting_heuristic(*args, **kwargs):
            heuristic_call_count.append(1)
            return original_heuristic(*args, **kwargs)

        def counting_successors(*args, **kwargs):
            results = original_successors(*args, **kwargs)
            successor_item_count.append(len(results))
            return results

        with patch.object(core, "heuristic", side_effect=counting_heuristic), patch.object(
            core, "successors", side_effect=counting_successors
        ):
            result = core.solve(board, weight=1.0, max_nodes=1_000, timeout_seconds=1.0)

        self.assertTrue(result.solved)
        self.assertLessEqual(len(heuristic_call_count), 1)

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

    def test_dead_cells_for_o_unaffected_by_occupancy_penalty(self):
        static, _state, _board, _player = parse_board(
            [
                ["B", "B", "B", "B"],
                ["B", "U", "", "B"],
                ["B", "", "O", "B"],
                ["B", "B", "B", "B"],
            ]
        )

        # No win lines fit inside this fully-walled 2x2 floor, so the static,
        # occupancy-independent dead-cell set is every floor cell -- unchanged
        # by the occupancy-aware heuristic penalty added on top of it.
        self.assertEqual(static.dead_cells_for_o, static.floor)

    def test_deadlock_prunes_o_pair_with_no_common_possible_line(self):
        static, state, _board, _player = parse_board(
            [
                ["U", "O", ""],
                ["B", "B", "B"],
                ["", "O", ""],
            ]
        )

        self.assertFalse(any(cell in static.dead_cells_for_o for cell in state.os))
        self.assertTrue(is_deadlock(state.os, state.xs, static))

    def _corridor_heuristic(self, xs_cells):
        static, state, _board, _player = parse_board([["U", "O", "", "", "", "O"]])
        blocked_state = State(player=state.player, os=state.os, xs=frozenset(xs_cells))
        return heuristic(blocked_state, static)

    def test_line_plan_primary_heuristic_does_not_count_unreachable_player_target(self):
        self.assertEqual(self._corridor_heuristic([]), 2.0)

    def test_occupancy_penalty_increases_when_x_blocks_box_path(self):
        self.assertGreater(self._corridor_heuristic([2]), self._corridor_heuristic([]))

    def test_occupancy_penalty_increases_when_x_blocks_stand_cell(self):
        self.assertGreater(self._corridor_heuristic([0]), self._corridor_heuristic([]))

    def test_occupancy_penalty_does_not_leak_across_assignments(self):
        self.assertEqual(self._corridor_heuristic([4]), 4.5)

    def test_occupancy_penalty_ignores_x_off_any_route(self):
        static, state, _board, _player = parse_board(
            [
                ["U", "O", "", "", "", "O", "B"],
                ["", "", "", "", "", "", "X"],
            ]
        )

        self.assertEqual(heuristic(state, static), 2.0)

    def test_occupancy_penalty_never_produces_infinite_heuristic_from_x_alone(self):
        static, state, _board, _player = parse_board([["U", "O", "X", "X", "", "O"]])

        self.assertTrue(math.isfinite(heuristic(state, static)))

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

    def test_solve_v1_remains_available(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]

        result = solve_v1(board, weight=1.0, max_nodes=1_000, timeout_seconds=1.0)

        self.assertTrue(result.solved)
        self.assertEqual(result.strategy, "v1_weighted")

    def test_package_solve_uses_verified_v2_portfolio_result(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]

        result = solve(board, weight=1.0, max_nodes=1_000, timeout_seconds=1.0)
        verification = verify_solution(board, result.moves)

        self.assertTrue(result.solved)
        self.assertTrue(verification.ok, verification.error)
        self.assertEqual(result.strategy, "v1_weighted")
        self.assertGreaterEqual(len(result.attempts), 1)

    def test_rank_discrepancy_strategy_smoke_test(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)

        result = core._run_strategy(
            context,
            config=SearchStrategyConfig(
                name="rank_discrepancy",
                kind="rank_discrepancy",
                bias_scale=1.25,
            ),
            max_nodes=1_000,
            deadline=None,
        )

        self.assertTrue(result.solved)
        self.assertEqual(result.strategy, "rank_discrepancy")

    def test_macro_children_expand_to_normal_push_sequence(self):
        board = [["U", "X", "", "", "O", "O"]]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)
        children = core._strategy_children_for(
            context,
            state,
            use_macros=True,
            bias_scale=1.0,
        )

        self.assertTrue(any(len(pushes) > 1 for pushes, *_rest in children))

    def test_macro_strategy_returns_verified_moves(self):
        board = [["U", "O", "", "", "O", ""]]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)

        result = core._run_strategy(
            context,
            config=SearchStrategyConfig(
                name="macro_greedy",
                kind="weighted",
                weight=2.5,
                g_weight=0.20,
                bias_scale=1.5,
                use_macros=True,
            ),
            max_nodes=1_000,
            deadline=None,
        )
        verification = verify_solution(board, result.moves)

        self.assertTrue(result.solved)
        self.assertTrue(verification.ok, verification.error)

    def test_portfolio_short_circuits_initial_x_loss(self):
        result = solve(
            [
                ["U", "", "", ""],
                ["X", "X", "X", ""],
                ["O", "", "O", ""],
            ],
            max_nodes=100,
            timeout_seconds=1.0,
        )

        self.assertFalse(result.solved)
        self.assertEqual(result.failure_reason, "x_loss")
        self.assertEqual(result.strategy, "precheck")
        self.assertEqual(result.attempts, ())

    def test_portfolio_short_circuits_initial_deadlock(self):
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
        self.assertEqual(result.failure_reason, "deadlock")
        self.assertEqual(result.strategy, "precheck")
        self.assertEqual(result.attempts, ())

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
