import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from solver.push_solver import rank_policy
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
    solve_v2,
    successors,
)
from solver.push_solver.verify import verify_solution
from solver.push_solver.policy_features import features_for_child
from solver.push_solver.rank_policy import default_policy
from solver.push_solver.verify import VerificationResult


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

    def test_frozen_pieces_detects_wall_locked_x(self):
        static, state, _board, _player = parse_board(
            [
                ["B", "B", "B", "B"],
                ["B", "U", "X", "B"],
                ["B", "B", "B", "B"],
            ]
        )

        frozen_os, frozen_xs = core.frozen_pieces(state.os, state.xs, static)

        self.assertEqual(frozen_os, frozenset())
        self.assertEqual(frozen_xs, frozenset({static.index(1, 2)}))

    def test_frozen_o_on_viable_line_is_not_deadlock(self):
        static, state, _board, _player = parse_board(
            [
                ["B", "B", "B", "B", "B"],
                ["B", "U", "O", "O", "B"],
                ["B", "B", "B", "B", "B"],
            ]
        )

        frozen_os, _frozen_xs = core.frozen_pieces(state.os, state.xs, static)

        self.assertIn(static.index(1, 3), frozen_os)
        self.assertFalse(
            is_deadlock(
                state.os,
                state.xs,
                static,
                player=state.player,
            )
        )

    def test_frozen_x_constraints_can_eliminate_all_lines(self):
        static, state, _board, _player = parse_board([["U", "O", "X", "O", ""]])

        self.assertFalse(
            core._has_viable_line_under_frozen_constraints(
                state.os,
                frozenset(),
                frozenset({static.index(0, 2)}),
                static,
                player=state.player,
            )
        )

    def test_x_aware_deadlock_does_not_prune_known_hard_solution_paths(self):
        from solver.push_solver.training_export import (
            decode_board,
            load_boards,
            load_solutions,
            push_path,
        )

        board_ids = {"20250928", "20251005", "20251116", "20251221"}
        boards = load_boards()
        solutions = {
            str(row["id"]): row.get("solution")
            for row in load_solutions()
            if str(row.get("id")) in board_ids
        }

        for board_id in sorted(board_ids):
            board = decode_board(boards[board_id])
            static, _state, _normalized, _player = parse_board(board)
            for state, _push in push_path(board, str(solutions[board_id])):
                self.assertFalse(
                    is_deadlock(
                        state.os,
                        state.xs,
                        static,
                        player=state.player,
                    ),
                    board_id,
                )

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

    def test_verifier_builds_static_lines_once_per_replay(self):
        from solver.push_solver import verify

        board = [["U", "", "O", "O"]]
        with patch.object(verify, "_valid_lines", wraps=verify._valid_lines) as build_lines:
            verification = verify_solution(board, "R")

        self.assertTrue(verification.ok, verification.error)
        self.assertEqual(build_lines.call_count, 1)

    def test_verifier_rejects_x_loss_created_by_push(self):
        verification = verify_solution([["X", "X", "", "X", "U"]], "L")

        self.assertFalse(verification.ok)
        self.assertEqual(verification.error, "x_loss:1")

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

    def test_solve_v2_remains_available(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]

        result = solve_v2(board, weight=1.0, max_nodes=1_000, timeout_seconds=1.0)

        self.assertTrue(result.solved)
        self.assertEqual(result.strategy, "v1_weighted")

    def test_package_solve_uses_verified_v3_result(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]

        result = solve(board, weight=1.0, max_nodes=1_000, timeout_seconds=1.0)
        verification = verify_solution(board, result.moves)

        self.assertTrue(result.solved)
        self.assertTrue(verification.ok, verification.error)
        self.assertIn(result.strategy, {"v1_weighted", "v3_keystroke_anytime"})
        self.assertIsNotNone(result.baseline_keystrokes)
        self.assertLessEqual(len(result.moves), result.baseline_keystrokes)
        self.assertGreaterEqual(len(result.attempts), 1)

    def test_v3_preserves_v2_incumbent_when_optimizer_errors(self):
        from solver.push_solver import portfolio

        board = [["U", "", "O", "O"]]
        baseline = core.PushSolveResult(
            solved=True,
            moves="R",
            final_board=(("", "U", "O", "O"),),
            pushes=(),
            nodes_expanded=7,
            peak_closed_size=7,
            elapsed_ms=1.0,
            failure_reason=None,
            strategy="v1_weighted",
        )

        with patch.object(portfolio, "solve_v2", return_value=baseline), patch(
            "solver.push_solver.optimizer.improve_solution",
            side_effect=RuntimeError("quality failure"),
        ):
            result = portfolio.solve(board, max_nodes=100, timeout_seconds=1.0)

        self.assertTrue(result.solved)
        self.assertEqual(result.moves, baseline.moves)
        self.assertEqual(result.final_board, baseline.final_board)
        self.assertEqual(result.baseline_keystrokes, 1)
        self.assertIn("optimizer_error", result.attempts[-1].failure_reason)

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
        board = [["U", "O", "", "", "O", ""]]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)
        children = core._strategy_children_for(
            context,
            state,
            use_macros=True,
            bias_scale=1.0,
            policy_weight=0.0,
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

    def test_committed_beam_strategy_returns_verified_moves(self):
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
                name="committed_beam",
                kind="committed_beam",
                beam_restart_widths=(8,),
                beam_restart_depths=(8,),
                beam_plan_limit=4,
            ),
            max_nodes=1_000,
            deadline=None,
        )
        verification = verify_solution(board, result.moves)

        self.assertTrue(result.solved)
        self.assertEqual(result.strategy, "committed_beam")
        self.assertTrue(verification.ok, verification.error)

    def test_committed_beam_honors_node_cap(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)

        result = core._run_strategy(
            context,
            config=SearchStrategyConfig(name="committed_beam", kind="committed_beam"),
            max_nodes=0,
            deadline=None,
        )

        self.assertFalse(result.solved)
        self.assertEqual(result.failure_reason, "node_cap")

    def test_committed_beam_honors_timeout(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)

        result = core._run_strategy(
            context,
            config=SearchStrategyConfig(name="committed_beam", kind="committed_beam"),
            max_nodes=1_000,
            deadline=0.0,
        )

        self.assertFalse(result.solved)
        self.assertEqual(result.failure_reason, "timeout")

    def test_portfolio_includes_line_committed_strategies(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)
        configs = core._portfolio_configs(2.0, context=context)
        line_configs = [
            config
            for config, _fraction in configs
            if config.name.startswith("line_commit_")
        ]

        self.assertGreaterEqual(len(line_configs), 1)
        self.assertTrue(all(config.committed_plan is not None for config in line_configs))
        self.assertTrue(all(config.relevance_filter for config in line_configs))

    def test_non_final_portfolio_strategies_leave_time_for_fallbacks(self):
        configs = core._portfolio_configs(2.0)

        self.assertTrue(all(fraction < 1.0 for _config, fraction in configs[:-1]))
        self.assertEqual(configs[-1][0].name, "committed_beam")
        self.assertEqual(configs[-1][1], 1.0)

    def test_portfolio_ends_with_committed_beam_fallback(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)
        configs = core._portfolio_configs(2.0, context=context)

        self.assertEqual(configs[-1][0].name, "committed_beam")
        self.assertEqual(configs[-1][0].kind, "committed_beam")

    def test_portfolio_routes_tiny_open_region_to_greedy_coverage_recovery(self):
        from solver.push_solver.training_export import decode_board, load_boards

        board = decode_board(load_boards()["20260814"])
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)

        configs = core._portfolio_configs(2.0, context=context)

        self.assertEqual(configs[0][0].name, "greedy_low_g_recovery_first")
        self.assertEqual(configs[0][1], 0.20)

    def test_portfolio_routes_anchor_shape_to_deep_v1_coverage_recovery(self):
        from solver.push_solver.training_export import decode_board, load_boards

        board = decode_board(load_boards()["20260523"])
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)

        configs = core._portfolio_configs(2.0, context=context)

        self.assertEqual(configs[0][0].name, "v1_deep_recovery_first")
        self.assertEqual(configs[0][1], 1.00)

    def test_portfolio_rejects_invalid_verified_solution(self):
        from solver.push_solver import portfolio

        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]

        def invalid_strategy(*_args, **_kwargs):
            return core.PushSolveResult(
                solved=True,
                moves="",
                final_board=None,
                pushes=(),
                nodes_expanded=1,
                peak_closed_size=1,
                elapsed_ms=0.0,
                failure_reason=None,
                strategy="fake",
            )

        with patch.object(portfolio, "_run_strategy", side_effect=invalid_strategy), patch(
            "solver.push_solver.verify.verify_solution",
            return_value=VerificationResult(ok=False, final_board=None, error="bad"),
        ):
            result = solve(board, max_nodes=20, timeout_seconds=1.0)

        self.assertFalse(result.solved)
        self.assertIn("invalid_solution", result.attempts[-1].failure_reason)

    def test_default_rank_policy_scores_legal_child_features(self):
        board = [
            ["", "", "", ""],
            ["U", "O", "", ""],
            ["", "", "O", ""],
        ]
        static, state, _normalized, initial_player = parse_board(board)
        context = core._build_search_context(static, state, initial_player)
        push, child, child_region, child_h, bias = core._successors_for(context, state)[0]
        features = features_for_child(
            context,
            state,
            (push,),
            child,
            child_region,
            child_h,
            bias,
            1,
        )
        policy = default_policy()

        self.assertIsNotNone(policy)
        self.assertIn("linear_push_policy_backfill_v1", policy.name)
        self.assertIsInstance(policy.score(features), float)
        self.assertIsInstance(policy.raw_score(features), float)
        self.assertIsInstance(policy.value(features), float)

    def test_default_rank_policy_honors_configured_primary_path(self):
        with TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "candidate.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "name": "candidate-policy",
                        "weights": {"child_h": 1.5},
                        "value_weights": {},
                        "state_action_hints": {},
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {"PUSH_RANK_POLICY_PATH": str(policy_path)},
                ),
                patch.object(rank_policy, "OPTIONAL_POLICY_PATHS", ()),
            ):
                rank_policy._DEFAULT_POLICY = None
                rank_policy._DEFAULT_POLICY_PATHS = None
                policy = default_policy()

        rank_policy._DEFAULT_POLICY = None
        rank_policy._DEFAULT_POLICY_PATHS = None
        self.assertEqual(policy.name, "candidate-policy")
        self.assertEqual(policy.raw_score({"child_h": 2.0}), 3.0)

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
