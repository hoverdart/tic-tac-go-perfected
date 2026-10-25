"""Independent legality, cache, and cooperative search checks for V4."""

import random
import itertools
import time
import unittest
from collections import deque
from unittest.mock import patch

from solver.push_solver import solve_v4, verify_solution
from solver.push_solver.v4 import Context, Session, mask_of
from solver.push_solver.state import parse_board
from solver.push_solver.search import _build_search_context, _successors_for
from solver.legacy_solver import apply_single_move, solved, lost_check


def consume(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as result:
            return result.value


def context(board):
    ctx = Context(board, time.perf_counter() + 30, 500000, route_limit=1)
    consume(ctx.initialize())
    return ctx


class PushSolverV4Test(unittest.TestCase):
    def test_bubble_wrap_gets_early_rank_search_without_action_hints(self):
        from datetime import date
        from apps.api.puzzle_catalog import CatalogEntry
        from solver.push_solver.benchmark_v4 import corpus

        row = next(r for r in corpus() if r["id"] == "20261025")
        board = CatalogEntry(
            date(2026, 10, 25), row["name"], row["width"], row["height"], row["puzzle"]
        ).board()
        with patch(
            "solver.push_solver.rank_policy.default_policy",
            side_effect=AssertionError("exact hints accessed"),
        ):
            result = solve_v4(board, timeout_seconds=15, quality_timeout_seconds=0)
        self.assertTrue(result.solved)
        self.assertTrue(verify_solution(board, result.moves).ok)
        self.assertEqual(result.attempts[0].strategy, "rank_discrepancy")
        counters = result.diagnostics["counters"]
        self.assertGreater(counters["reverse_hits"] / counters["reverse_queries"], 0.9)

    def test_all_sessions_resume_and_share_one_expansion_counter(self):
        from solver.push_solver.v4 import STRATEGIES, BudgetExpired

        board = [
            ["U", "", "", ""],
            ["", "X", "O", ""],
            ["", "O", "", ""],
            ["", "", "", ""],
        ]
        for kind in STRATEGIES:
            ctx = context(board)
            session = Session(ctx, kind)
            session.advance(quantum=0, expansions=1)
            for _ in range(1000):
                session.advance(quantum=0.001, expansions=1)
                if session.done:
                    break
            self.assertTrue(session.done, kind)
            self.assertIsNotNone(session.found, kind)
            self.assertTrue(
                verify_solution(board, ctx.reconstruct(session.found)[0]).ok
            )
        ctx = context(board)
        ctx.max_nodes = 2
        sessions = [Session(ctx, kind) for kind in STRATEGIES]
        with self.assertRaises(BudgetExpired):
            for session in sessions:
                session.advance(quantum=1, expansions=1)
        self.assertEqual(ctx.expanded, 2)
        self.assertEqual(sum(s.nodes for s in sessions), 2)

    def test_alternate_routes_include_a_longer_detour_and_remain_deterministic(self):
        board = [
            ["U", "", "", "", ""],
            ["", "", "", "", ""],
            ["", "O", "", "O", ""],
            ["", "", "", "", ""],
            ["", "", "", "", ""],
        ]
        a = context(board)
        a.route_limit = 3
        paths = consume(a.routes(6, 8))
        self.assertGreater(len(paths), 1)
        self.assertEqual(paths[0][0], 2)
        self.assertTrue(all(r[0] <= 4 for r in paths))
        self.assertTrue(any(r[0] > 2 for r in paths))
        self.assertEqual(paths, consume(a.routes(6, 8)))

    def test_complete_short_plan_result_is_cached(self):
        ctx = context([["U", "O", "", "O"]])
        first = consume(ctx.score(ctx.start))
        computations = ctx.metrics["score_computations"]
        self.assertLess(len(first[1]), 24)
        self.assertEqual(consume(ctx.score(ctx.start)), first)
        self.assertEqual(ctx.metrics["score_computations"], computations)

    def test_quality_timeout_preserves_verified_incumbent_and_total_budget(self):
        board = [
            ["U", "", "", ""],
            ["", "X", "O", ""],
            ["", "O", "", ""],
            ["", "", "", ""],
        ]
        with patch(
            "solver.push_solver.v4.improve_keystrokes",
            side_effect=RuntimeError("quality interrupted"),
        ):
            result = solve_v4(board, timeout_seconds=0.1)
        self.assertTrue(result.solved)
        self.assertTrue(verify_solution(board, result.moves).ok)
        self.assertLess(result.elapsed_ms, 150)
        self.assertLessEqual(result.nodes_expanded, 500000)

    def test_correct_paths_and_multi_o_dead_cell(self):
        for board in [
            [["U", "", ""], ["O", "O", ""], ["", "", ""]],
            [["O", "B", "U", "O", "", "O", ""]],
            [["U", "", "O", "O", "O"]],
        ]:
            with self.subTest(board=board):
                r = solve_v4(board, timeout_seconds=2, quality_timeout_seconds=0)
                self.assertTrue(r.solved)
                self.assertTrue(verify_solution(board, r.moves).ok)
                self.assertFalse(r.diagnostics["exact_hints"])

    def test_reverse_map_cache_is_scoped_by_board_blockers_and_target(self):
        a = context([["U", "", ""], ["O", "O", ""], ["", "", ""]])
        for blockers in [a.walls, a.walls | (1 << 4)]:
            first = consume(a.reverse_map(blockers, 4))
            again = consume(a.reverse_map(blockers, 4))
            self.assertEqual(first, again)
        self.assertEqual(a.metrics["reverse_computations"], 2)
        self.assertEqual(a.metrics["reverse_hits"], 2)
        b = context([["U", "", ""], ["O", "O", "B"], ["", "", ""]])
        self.assertEqual(len(b.reverse_cache), 0)

    def test_partial_successor_generation_resumes_without_publishing_a_partial_cache(
        self,
    ):
        ctx = context(
            [["U", "", "", ""], ["", "X", "O", ""], ["", "O", "", ""], ["", "", "", ""]]
        )
        first = ctx.successors(ctx.start)
        for _ in range(10):
            next(first)
        second = [c for c in ctx.successors(ctx.start) if c is not None]
        remainder = [c for c in first if c is not None]
        self.assertEqual(second, remainder)
        self.assertTrue(ctx.successor_cache[ctx.start][2])
        self.assertEqual(ctx.metrics["successor_computations"], 1)

    def test_resumable_session_keeps_frontier_and_obeys_shared_node_cap(self):
        board = [
            ["U", "", "", ""],
            ["", "X", "O", ""],
            ["", "O", "", ""],
            ["", "", "", ""],
        ]
        ctx = context(board)
        session = Session(ctx, "rank_discrepancy")
        session.advance(quantum=0, expansions=1)
        before = ctx.metrics["generated"]
        for _ in range(1000):
            session.advance(quantum=0.01, expansions=1)
            if session.done:
                break
        self.assertTrue(session.done)
        self.assertGreaterEqual(ctx.metrics["generated"], before)
        if session.found:
            moves, _ = ctx.reconstruct(session.found)
            self.assertTrue(verify_solution(board, moves).ok)
        capped = solve_v4(
            board, max_nodes=1, timeout_seconds=1, quality_timeout_seconds=0
        )
        self.assertLessEqual(capped.nodes_expanded, 1)
        expired = solve_v4(board, timeout_seconds=0, quality_timeout_seconds=0)
        self.assertFalse(expired.solved)
        self.assertEqual(expired.failure_reason, "timeout")

    def test_two_o_successors_match_reference_on_reachable_positions(self):
        board = [
            ["U", "", "", ""],
            ["", "X", "O", ""],
            ["", "O", "", ""],
            ["", "", "", ""],
        ]
        static, state, _, player = parse_board(board)
        old = _build_search_context(static, state, player)
        new = context(board)
        pending = deque([(state, new.start)])
        visited = set()
        while pending and len(visited) < 80:
            previous, current = pending.popleft()
            if previous in visited:
                continue
            visited.add(previous)
            if new.goal(current)[0]:
                continue
            ref = _successors_for(old, previous)
            actual = consume(new.children(current))
            refkeys = {
                (p.piece, p.cell, p.move): (s, region, h, bias)
                for p, s, region, h, bias in ref
            }
            keys = {
                (c.pushes[0].piece, c.pushes[0].cell, c.pushes[0].move): c
                for c in actual
            }
            self.assertEqual(set(refkeys), set(keys))
            for k, (s, region, h, bias) in refkeys.items():
                child = keys[k]
                self.assertEqual(
                    child.state, (mask_of(s.os), mask_of(s.xs), mask_of(region))
                )
                self.assertAlmostEqual(child.h, h)
                self.assertAlmostEqual(child.bias, bias)
                pending.append((s, child.state))

    def test_exhaustive_small_board_oracle_has_no_false_deadlocks(self):
        rng = random.Random(4)
        for sample in range(75):
            cells9 = (
                ["U", "O", "O", "O", "X", "B", "", "", ""]
                if sample % 2
                else ["U", "O", "O", "X", "X", "", "", "", ""]
            )
            rng.shuffle(cells9)
            board = tuple(tuple(cells9[r * 3 : r * 3 + 3]) for r in range(3))
            queue = deque([board])
            seen = {board}
            possible = False
            while queue:
                current = queue.popleft()
                if lost_check(current):
                    continue
                if solved(current):
                    possible = True
                    break
                for move in "UDLR":
                    child = apply_single_move(current, move)
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)
            result = solve_v4(board, timeout_seconds=2, quality_timeout_seconds=0)
            self.assertEqual(result.solved, possible, board)
            if possible:
                self.assertTrue(verify_solution(board, result.moves).ok)

    def test_every_small_two_o_wall_and_x_arrangement_against_exhaustive_oracle(self):
        for layout in sorted(
            set(itertools.permutations(("U", "O", "O", "X", "B", "")))
        ):
            board = (layout[:3], layout[3:])
            queue = deque([board])
            seen = {board}
            possible = False
            while queue:
                current = queue.popleft()
                if lost_check(current):
                    continue
                if solved(current):
                    possible = True
                    break
                for move in "UDLR":
                    child = apply_single_move(current, move)
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)
            result = solve_v4(board, timeout_seconds=1, quality_timeout_seconds=0)
            self.assertEqual(result.solved, possible, board)
            if result.solved:
                self.assertTrue(verify_solution(board, result.moves).ok)


if __name__ == "__main__":
    unittest.main()
