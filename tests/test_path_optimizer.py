import time
import unittest
from unittest.mock import patch

from solver.push_solver import solve_v4, verify_solution
from solver.push_solver.models import Push
from solver.push_solver.path_optimizer import (
    PathRewriter,
    RewriteExpired,
    improve_vicinity,
)
from solver.push_solver.v4 import Context


class PathOptimizerTest(unittest.TestCase):
    def test_unnecessary_x_work_is_replaced_by_a_shorter_winning_walk(self):
        board = [[""] * 5 for _ in range(4)]
        board[3][0], board[1][1], board[1][3], board[2][3] = "U", "O", "O", "X"
        ctx = Context(board, None, None)
        original = "URRRLU"
        self.assertTrue(verify_solution(board, original).ok)
        result = PathRewriter(ctx, time.perf_counter() + 1).optimize(
            original, (Push("X", 13, "R"),)
        )
        self.assertEqual(len(result.moves), 4)
        self.assertEqual(result.pushes, ())
        self.assertTrue(verify_solution(board, result.moves).ok)

    def test_closest_winning_target_is_used_instead_of_first_line(self):
        board = [[""] * 5 for _ in range(3)]
        board[0][4], board[1][1], board[1][2] = "U", "O", "O"
        ctx = Context(board, None, None)
        original = ctx.walk(ctx.player, 5, ctx.os | ctx.xs)
        self.assertEqual(len(original), 5)
        result = PathRewriter(ctx, time.perf_counter() + 1).optimize(original, ())
        self.assertEqual(len(result.moves), 2)
        self.assertTrue(verify_solution(board, result.moves).ok)

    def test_reordering_cannot_cross_an_intermediate_x_loss(self):
        board = [[""] * 5 for _ in range(5)]
        board[3][2], board[2][0], board[4][4] = "U", "O", "O"
        board[0][0], board[0][1], board[1][2] = "X", "X", "X"
        ctx = Context(board, None, None)
        rewriter = PathRewriter(ctx, time.perf_counter() + 1)
        label = rewriter.initial.index(7)
        self.assertIsNone(rewriter.replay(((label, "U"),), 100))

    def test_ooo_win_can_finish_without_walking_or_pushing(self):
        board = [["U", "", "O", "O", "O"]]
        result = PathRewriter(Context(board, None, None), time.perf_counter() + 1).optimize("R", ())
        self.assertEqual(result.moves, "")
        self.assertTrue(verify_solution(board, result.moves).ok)

    def test_deadline_preserves_a_verified_improvement(self):
        board = [["U", "", "", ""], ["", "X", "O", ""], ["", "O", "", ""], ["", "", "", ""]]
        baseline = solve_v4(board, timeout_seconds=1, quality_timeout_seconds=0)
        self.assertTrue(baseline.solved)
        original = "RLRL" + baseline.moves
        self.assertTrue(verify_solution(board, original).ok)
        rewriter = PathRewriter(Context(board, None, None), time.perf_counter() + 1)
        replay = rewriter.replay
        calls = 0

        def expire_after_first_improvement(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RewriteExpired
            return replay(*args)

        with patch.object(rewriter, "replay", side_effect=expire_after_first_improvement):
            result = rewriter.optimize(original, baseline.pushes)
        self.assertGreater(calls, 1)
        self.assertLess(len(result.moves), len(original))
        self.assertTrue(verify_solution(board, result.moves).ok)

    def test_expired_optimizer_returns_the_original_path(self):
        board = [["U", "O", "", "O"]]
        original = "R"
        result = PathRewriter(Context(board, None, None), time.perf_counter() - 1).optimize(
            original, (Push("O", 1, "R"),)
        )
        self.assertEqual(result.moves, original)
        self.assertTrue(verify_solution(board, result.moves).ok)

    def test_vicinity_can_replace_a_push_direction_that_reordering_cannot_change(self):
        board = [["X", "B", "U", ""], ["", "", "O", ""], ["", "O", "", ""], ["X", "", "", ""]]
        original = "RDDLULLDRUR"
        pushes = (Push("O", 6, "U"), Push("O", 9, "R"))
        ctx = Context(board, None, None)
        rewritten = PathRewriter(ctx, time.perf_counter() + 1).optimize(original, pushes)
        self.assertEqual(rewritten.moves, original)
        candidates = list(improve_vicinity(ctx, original, pushes, time.perf_counter() + 1))
        self.assertTrue(candidates)
        self.assertEqual(len(candidates[-1][0]), 3)
        self.assertTrue(verify_solution(board, candidates[-1][0]).ok)
        self.assertEqual(ctx.expanded, ctx.metrics["vicinity_expansions"])

    def test_vicinity_respects_the_shared_node_cap(self):
        board = [["X", "B", "U", ""], ["", "", "O", ""], ["", "O", "", ""], ["X", "", "", ""]]
        ctx = Context(board, None, 1)
        list(improve_vicinity(
            ctx, "RDDLULLDRUR", (Push("O", 6, "U"), Push("O", 9, "R")), time.perf_counter() + 1
        ))
        self.assertEqual(ctx.expanded, 1)
        self.assertEqual(ctx.metrics["vicinity_expansions"], 1)

    def test_vicinity_yields_verified_improvements_before_interruption(self):
        board = [[""] * 5 for _ in range(3)]
        board[0][4], board[1][1], board[1][2] = "U", "O", "O"
        ctx = Context(board, None, None)
        iterator = improve_vicinity(ctx, ctx.walk(ctx.player, 5, ctx.os), (), time.perf_counter() + 1)
        moves, _ = next(iterator)
        self.assertEqual(len(moves), 2)
        self.assertTrue(verify_solution(board, moves).ok)
        with patch("solver.push_solver.path_optimizer.time.perf_counter", return_value=float("inf")):
            self.assertEqual(list(iterator), [])

    def test_quality_phases_preserve_coverage_and_share_the_solver_node_cap(self):
        board = [["U", "", "", ""], ["", "X", "O", ""], ["", "O", "", ""], ["", "", "", ""]]
        result = solve_v4(board, max_nodes=64, timeout_seconds=0.2, quality_timeout_seconds=0.2)
        self.assertTrue(verify_solution(board, result.moves).ok)
        self.assertLessEqual(len(result.moves), result.baseline_keystrokes)
        self.assertLessEqual(result.nodes_expanded, 64)
        counters = result.diagnostics["counters"]
        self.assertEqual(
            result.quality_nodes_expanded,
            counters.get("vicinity_expansions", 0) + counters.get("keystroke_search_expansions", 0),
        )
