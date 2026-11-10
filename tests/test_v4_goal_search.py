"""Strategy changes preserve scoring semantics and expose every commitment."""

import json
import random
import time
import unittest
from dataclasses import asdict
from pathlib import Path

from solver.push_solver.v4 import (
    BudgetExpired,
    Context,
    GoalSession,
    Plan,
    Session,
    cells,
    mask_of,
)


def consume(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as result:
            return result.value


def context(board, routes=3):
    ctx = Context(board, time.perf_counter() + 30, 500000, route_limit=routes)
    consume(ctx.initialize())
    return ctx


class GoalSearchTest(unittest.TestCase):
    def test_complete_macro_cache_is_isolated_and_never_publishes_partial_work(self):
        board = [[""] * 6 for _ in range(6)]
        board[0][0], board[1][1], board[3][2], board[2][3] = "U", "O", "O", "X"
        ctx = context(board)
        ctx.goal_portfolio = True
        first = ctx.children(ctx.start, True)
        next(first)
        self.assertFalse(ctx.children_cache)
        second = consume(ctx.children(ctx.start, True))
        self.assertTrue(second)
        self.assertEqual(consume(first), second)
        expected = list(second)
        second.clear()
        self.assertEqual(consume(ctx.children(ctx.start, True)), expected)
        self.assertGreater(ctx.metrics["complete_children_hits"], 0)
        primitives = consume(ctx.children(ctx.start, False))
        self.assertTrue(all(len(child.pushes) == 1 for child in primitives))
        ctx.deadline = time.perf_counter() - 1
        with self.assertRaises(BudgetExpired):
            consume(ctx.children(ctx.start, True))

    def test_batched_coverage_checkpoints_keep_forced_boundaries_and_expiry(self):
        ctx = context([["U", "O", "O"]])
        ctx.goal_portfolio = True
        ctx.checkpoints = 0
        self.assertEqual(sum(len(list(ctx.checkpoint())) for _ in range(32)), 2)
        self.assertEqual(len(list(ctx.checkpoint(force=True))), 1)
        ctx.deadline = time.perf_counter() - 1
        with self.assertRaises(BudgetExpired):
            list(ctx.checkpoint(force=True))

    def test_compiled_freeze_axes_match_the_reference_fixed_point(self):
        rng = random.Random(9126)
        for rows, cols in ((4, 4), (3, 7), (8, 8)):
            for _ in range(10):
                board = [["B" if rng.random() < 0.25 else "" for _ in range(cols)] for _ in range(rows)]
                for cell, value in zip(rng.sample(range(rows * cols), 3), ("U", "O", "O")):
                    board[cell // cols][cell % cols] = value
                ctx = Context(board, None, None)
                for _ in range(20):
                    occupied = mask_of(i for i in cells(ctx.floor) if rng.random() < 0.7)
                    frozen = 0
                    while True:
                        before = frozen
                        blockers = ctx.walls | frozen
                        for cell in cells(occupied & ~frozen):
                            if all(blockers & ((1 << stand) | (1 << destination)) for _, stand, destination in ctx.transitions[cell]):
                                frozen |= 1 << cell
                        if before == frozen:
                            break
                    self.assertEqual(ctx.frozen(occupied), frozen)

    def test_route_choice_cache_preserves_cost_and_first_alternative_ties(self):
        board = [[""] * 6 for _ in range(6)]
        board[0][0], board[1][1], board[3][2] = "U", "O", "O"
        ctx = context(board)
        alternatives = consume(ctx.routes(7, 21))
        rng = random.Random(1126)
        for _ in range(40):
            xs = mask_of(i for i in cells(ctx.floor & ~ctx.os) if rng.random() < 0.3)
            expected = min(alternatives, key=lambda route: route[0] + (xs & route[1]).bit_count() + bool(route[1] & (1 << 20)))
            self.assertEqual(consume(ctx.best_route(7, 21, 20, xs)), expected)
            self.assertEqual(consume(ctx.best_route(7, 21, 20, xs)), expected)
        self.assertGreater(ctx.metrics["route_choice_hits"], 0)

    def test_reachability_matches_independent_flood_fill_on_rectangular_boards(self):
        rng = random.Random(20261109)
        for rows, cols in ((1, 5), (5, 1), (2, 3), (3, 7), (8, 8)):
            for _ in range(12):
                board = [
                    ["B" if rng.random() < 0.25 else "" for _ in range(cols)]
                    for _ in range(rows)
                ]
                for cell, value in zip(rng.sample(range(rows * cols), 3), ("U", "O", "O")):
                    board[cell // cols][cell % cols] = value
                ctx = Context(board, None, None)
                occupied = mask_of(
                    i for i in range(rows * cols) if rng.random() < 0.35
                )
                for start in range(rows * cols):
                    pending = [start]
                    seen = set()
                    while pending:
                        current = pending.pop()
                        r, c = divmod(current, cols)
                        if current in seen or board[r][c] == "B" or occupied & (1 << current):
                            continue
                        seen.add(current)
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                pending.append(nr * cols + nc)
                    self.assertEqual(ctx.reachable(start, occupied), mask_of(seen))

    def test_commitment_score_does_not_reward_drifting_to_a_different_line(self):
        board = [[""] * 6 for _ in range(6)]
        board[5][5], board[1][1], board[3][4] = "U", "O", "O"
        ctx = context(board)
        plan = Plan(0, mask_of((25, 26, 27)), 26, ((7, 25), (22, 27)), 0, 0)
        before = ctx.start
        os = mask_of((7, 16))  # The second O moves away from the committed row.
        after = os, 0, ctx.reachable(35, os)
        self.assertLess(consume(ctx.score(after))[0], consume(ctx.score(before))[0])
        self.assertGreater(
            ctx.commitment_score(plan, after), ctx.commitment_score(plan, before)
        )

    def test_any_verified_goal_wins_even_outside_the_committed_line(self):
        ctx = context([["", "", "", ""], ["U", "O", "O", ""], ["", "", "", ""]])
        plan = Plan(0, mask_of((0, 1, 2)), 1, ((5, 0), (6, 2)), 0, 0)
        self.assertEqual(ctx.commitment_score(plan, ctx.start), 0)

    def test_coverage_lanes_share_one_node_budget_and_resume_each_frontier(self):
        from datetime import date

        from apps.api.puzzle_catalog import CatalogEntry
        from solver.push_solver.benchmark_v4 import corpus

        row = next(r for r in corpus() if r["id"] == "20260301")
        ctx = context(CatalogEntry(
            date.fromisoformat(row["date"]), row["name"],
            row["width"], row["height"], row["puzzle"],
        ).board())
        ctx.max_nodes = 128
        ctx.goal_portfolio = True
        parent = Session(ctx, "committed_beam")
        with self.assertRaises(BudgetExpired):
            for _ in range(10000):
                before = ctx.expanded
                parent.advance(quantum=0.005, expansions=1)
                self.assertLessEqual(ctx.expanded - before, 1)
        self.assertEqual(ctx.expanded, 128)
        self.assertEqual(ctx.expanded, parent.nodes)
        self.assertEqual(ctx.expanded, sum(g.nodes for g in parent.anchor_goals))
        self.assertEqual({g.objective for g in parent.anchor_goals}, {"global", "goal"})
        self.assertTrue(any(g.widths[0][0] == 96 for g in parent.anchor_goals))

    def test_lazy_prefixes_exactly_match_frozen_prechange_scoring(self):
        rows = json.loads(
            (Path(__file__).parent / "fixtures/v4-scoring-reference.json").read_text()
        )
        contexts = {}
        for row in rows:
            key = json.dumps(row["board"]), row["routes"]
            if key not in contexts:
                contexts[key] = context(row["board"], row["routes"])
            ctx = contexts[key]
            state = tuple(row["state"])
            for limit in (1, 8, 24, 1):
                h, plans = consume(ctx.score(state, limit=limit))
                self.assertEqual(h, row["h"])
                actual = json.loads(json.dumps([asdict(p) for p in plans]))
                self.assertEqual(actual, row["plans"][:limit])

    def test_interleaved_prefix_requests_publish_one_correct_prefix(self):
        ctx = context(
            [["U", "", "", ""], ["", "X", "O", ""], ["", "O", "", ""], ["", "", "", ""]]
        )
        large = ctx.score(ctx.start, limit=24)
        while ctx.metrics["score_computations"] == 0:
            next(large)
        first = consume(ctx.score(ctx.start, limit=1))
        full = consume(large)
        self.assertEqual(full[1][:1], first[1])
        self.assertEqual(len(full[1]), len(set(full[1])))
        self.assertEqual(ctx.metrics["score_computations"], 1)
        self.assertEqual(ctx.metrics["plans_materialized"], len(full[1]))

    def test_all_static_goal_lines_and_assignments_remain_in_commitment_pool(self):
        ctx = context(
            [["U", "", "", ""], ["", "X", "O", ""], ["", "O", "", ""], ["", "", "", ""]]
        )
        geometry = consume(ctx.geometry(ctx.os))
        plans = consume(ctx.beam_plans(ctx.start))
        self.assertEqual(len(plans), len(geometry))
        self.assertEqual({p.line for p in plans}, {g[0] for g in geometry})
        self.assertEqual(
            {(p.line, p.target, p.assignment) for p in plans},
            {(g[0], g[1], ((g[2], g[3]), (g[4], g[5]))) for g in geometry},
        )
        lines = len({p.line for p in plans})
        self.assertEqual(len({p.line for p in plans[:lines]}), lines)

    def test_goal_sessions_keep_separate_frontiers_and_shared_node_accounting(self):
        ctx = context(
            [["U", "", "", ""], ["", "X", "O", ""], ["", "O", "", ""], ["", "", "", ""]]
        )
        parent = Session(ctx, "committed_beam")
        plans = consume(ctx.beam_plans(ctx.start))
        a, b = [GoalSession(parent, p, i) for i, p in enumerate(plans[:2])]
        a.advance(expansions=1)
        b.advance(expansions=1)
        self.assertIsNot(a.iterator, b.iterator)
        self.assertEqual(ctx.expanded, parent.nodes)
        self.assertEqual(ctx.expanded, a.nodes + b.nodes)
        prior = a.nodes
        a.advance(quantum=0)
        self.assertGreaterEqual(a.nodes, prior)

    def test_nested_beam_slices_never_exceed_the_parent_expansion_limit(self):
        ctx = context(
            [["U", "", "", ""], ["", "X", "O", ""], ["", "O", "", ""], ["", "", "", ""]]
        )
        parent = Session(ctx, "committed_beam")
        for _ in range(1000):
            before = ctx.expanded
            parent.advance(quantum=0.005, expansions=1)
            self.assertLessEqual(ctx.expanded - before, 1)
            self.assertEqual(parent.nodes, ctx.expanded)
            if parent.done:
                break
        self.assertTrue(parent.done)

    def test_forced_macros_match_the_reference_and_keep_primitive_pushes(self):
        from solver.push_solver.models import Push, State
        from solver.push_solver.search import (
            _build_search_context,
            _same_piece_continuation,
        )
        from solver.push_solver.state import parse_board

        board = [
            ["U", "", "", ""],
            ["", "X", "O", ""],
            ["", "O", "", ""],
            ["", "", "", ""],
        ]
        ctx = context(board, routes=1)
        ctx.macro_mode = "forced"
        static, _, _, _ = parse_board(board)
        pending, seen = [ctx.start], set()
        continuations = 0
        while pending and len(seen) < 25:
            state = pending.pop(0)
            if state in seen:
                continue
            seen.add(state)
            primitives = consume(ctx.children(state))
            pending.extend(c.state for c in primitives)
            converted = State(
                next(cells(state[2])),
                frozenset(cells(state[0])),
                frozenset(cells(state[1])),
            )
            reference = _build_search_context(static, converted, converted.player)
            for piece, occupied in (("O", state[0]), ("X", state[1])):
                for cell in cells(occupied):
                    for move, _, _ in ctx.transitions[cell]:
                        expected = _same_piece_continuation(
                            reference, converted, piece=piece, cell=cell, move=move
                        )
                        actual = consume(
                            ctx.macro_continuation(state, Push(piece, cell, move))
                        )
                        self.assertEqual(actual is None, expected is None)
                        if expected is not None:
                            continuations += 1
                            self.assertEqual(actual.pushes[0], expected[0])
                            self.assertEqual(
                                actual.state[:2],
                                (mask_of(expected[1].os), mask_of(expected[1].xs)),
                            )
            self.assertEqual(consume(ctx.children(state)), primitives)
        self.assertGreater(continuations, 0)

    def test_commitment_filter_preserves_the_verified_winning_lines(self):
        from datetime import date

        from apps.api.puzzle_catalog import CatalogEntry
        from solver.push_solver.benchmark_v4 import TARGETS
        from solver.push_solver.verify import verify_solution

        path = (
            Path(__file__).parents[1]
            / "debug-artifacts/v4-2026-10-25/verified-path-corpus.jsonl"
        )
        ids = set(TARGETS) | {"20250928", "20260301", "20260802", "20260823"}
        for row in map(json.loads, path.read_text().splitlines()):
            if row["id"] not in ids:
                continue
            board = CatalogEntry(
                date.fromisoformat(row["date"]),
                row["name"],
                row["width"],
                row["height"],
                row["puzzle"],
            ).board()
            replay = verify_solution(board, row["solution"])
            self.assertTrue(replay.ok, row["id"])
            final = [cell for line in replay.final_board for cell in line]
            ctx = context(board)
            winning = {
                line
                for line in ctx.lines
                if all(final[cell] in ("O", "U") for cell in cells(line))
            }
            self.assertTrue(
                winning & {p.line for p in consume(ctx.beam_plans(ctx.start))},
                row["id"],
            )

    def test_unused_frozen_o_does_not_remove_other_goal_commitments(self):
        ctx = context(
            [
                ["O", "B", "", "", ""],
                ["B", "", "", "O", ""],
                ["", "U", "O", "", ""],
                ["", "", "", "", ""],
            ]
        )
        plans = consume(ctx.beam_plans(ctx.start))
        self.assertTrue(plans)
        self.assertTrue(
            any(all(source != 0 for source, _ in p.assignment) for p in plans)
        )
