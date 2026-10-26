"""Strategy changes preserve scoring semantics and expose every commitment."""

import json
import time
import unittest
from dataclasses import asdict
from pathlib import Path

from solver.push_solver.v4 import Context, GoalSession, Session, cells, mask_of


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
