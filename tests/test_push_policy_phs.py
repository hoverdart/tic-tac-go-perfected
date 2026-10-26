"""PHS probability, graph dominance, resume, and offline-training contracts."""

import json
import math
import tempfile
import time
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from solver.push_solver import solve_v4, verify_solution
from solver.push_solver.phs import dominates, log_priority
from solver.push_solver.push_policy import FEATURES, SCHEMA, PushPolicy, action_features
from solver.push_solver.train_push_policy import (
    board_for,
    consume,
    export_board,
    fit,
    split_identities,
)
from solver.push_solver.v4 import BudgetExpired, Child, Context, Session


class PushPolicyPHSTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "policy.json"
        self.payload = {
            "schema": SCHEMA,
            "name": "test",
            "weights": {},
            "temperature": 1.0,
            "uniform_mix": 0.05,
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.payload))
        return PushPolicy(self.path)

    def context(self, board):
        ctx = Context(board, time.perf_counter() + 10, 500000)
        consume(ctx.initialize())
        ctx.push_policy = PushPolicy(self.path)
        return ctx

    def test_stable_probabilities_normalize_and_keep_every_action(self):
        self.payload["weights"] = {FEATURES[0]: 1e200}
        policy = self.save()
        x = [(value,) + (0.0,) * (len(FEATURES) - 1) for value in (-1, 0, 1)]
        p = policy.probabilities(x)
        self.assertAlmostEqual(sum(p), 1)
        self.assertTrue(all(v >= policy.uniform_mix / 3 for v in p))
        self.assertGreater(p[2], p[0])
        self.assertEqual(policy.probabilities([]), ())
        self.assertEqual(policy.probabilities([x[0]]), (1.0,))
        with self.assertRaises(ValueError):
            policy.probabilities([(1.0,)])

    def test_strict_schema_rejects_hints_and_invalid_numbers(self):
        original = self.payload.copy()
        for override in [
            {"state_action_hints": {"secret": 10}},
            {"board_id": "remembered"},
            {"weights": {"unknown_feature": 1}},
            {"weights": {FEATURES[0]: float("nan")}},
            {"temperature": 0},
            {"temperature": float("inf")},
            {"uniform_mix": 0},
        ]:
            self.payload = original | override
            with self.assertRaises(ValueError):
                self.save()

    def test_log_priority_and_pareto_rules_preserve_probability_tradeoffs(self):
        self.assertAlmostEqual(log_priority(3, -math.log(0.01), 5), math.log(9 / 0.01))
        self.assertEqual(log_priority(3, 1, 5, ancestor=100), 100)
        self.assertAlmostEqual(
            log_priority(3, -math.log(0.01), 5, aggressive=True),
            math.log(9) - 2.25 * math.log(0.01),
        )
        self.assertTrue(math.isfinite(log_priority(500000, 4e6, math.inf)))
        self.assertTrue(dominates((1, 2, 3), (2, 3, 4)))
        self.assertFalse(dominates((1, 10, 3), (2, 1, 4)))
        self.assertFalse(dominates((1, 1, 10), (2, 2, 4)))

    def test_long_high_probability_and_short_low_probability_labels_both_expand(self):
        visits = Counter()
        graph = {
            "root": [("join", 0.1), ("a", 0.9)],
            "a": [("join", 1.0)],
            "join": [("root", 1.0)],
        }

        def checkpoint():
            yield None

        def score(state, **kwargs):
            yield None
            return 0.0, ()

        def children(state, *args):
            visits[state] += 1
            yield None
            return [
                Child((), nxt, 0, 0, probability) for nxt, probability in graph[state]
            ]

        def distribution(ctx, state, items):
            yield None
            return tuple(child.policy for child in items)

        ctx = SimpleNamespace(
            start="root",
            metrics=Counter(),
            expanded=0,
            max_nodes=100,
            peak_frontier=0,
            check=lambda: None,
            checkpoint=checkpoint,
            score=score,
            children=children,
            goal=lambda state: (False, None),
            push_policy=SimpleNamespace(distribution=distribution),
        )
        session = Session(ctx, "phs")
        for _ in range(100):
            before = session.nodes
            session.advance(quantum=1, expansions=1)
            self.assertLessEqual(session.nodes - before, 1)
            if session.done:
                break
        self.assertTrue(session.done)
        self.assertEqual(visits["join"], 2)
        self.assertEqual(visits["root"], 1)  # both cycles dominated by root label
        self.assertEqual(ctx.expanded, session.nodes)
        self.assertGreater(session.counters["pruned_policy_dominance"], 0)

    def test_real_search_resumes_and_replays_without_exact_hints(self):
        board = [
            ["U", "", "", ""],
            ["", "X", "O", ""],
            ["", "O", "", ""],
            ["", "", "", ""],
        ]
        for kind in ("phs", "levin", "phs_star"):
            ctx = self.context(board)
            session = Session(ctx, kind)
            for _ in range(10000):
                before = ctx.expanded
                session.advance(quantum=0.002, expansions=1)
                self.assertLessEqual(ctx.expanded - before, 1)
                if session.done:
                    break
            self.assertIsNotNone(session.found)
            self.assertTrue(
                verify_solution(board, ctx.reconstruct(session.found)[0]).ok
            )
        with patch(
            "solver.push_solver.rank_policy.default_policy",
            side_effect=AssertionError("hints used"),
        ):
            result = solve_v4(
                board, strategy="phs", policy_path=self.path, quality_timeout_seconds=0
            )
        self.assertTrue(result.solved)
        self.assertIn("push_policy", result.diagnostics["model_hashes"])
        self.assertFalse(result.diagnostics["exact_hints"])

    def test_deadline_and_shared_node_cap_include_policy_work(self):
        board = [
            ["U", "", "", ""],
            ["", "X", "O", ""],
            ["", "O", "", ""],
            ["", "", "", ""],
        ]
        result = solve_v4(
            board,
            strategy="phs",
            policy_path=self.path,
            timeout_seconds=0,
            quality_timeout_seconds=0,
        )
        self.assertFalse(result.solved)
        self.assertEqual(result.failure_reason, "timeout")
        ctx = self.context(board)
        ctx.max_nodes = 2
        sessions = [Session(ctx, kind) for kind in ("phs", "weighted")]
        with self.assertRaisesRegex(BudgetExpired, "node_cap"):
            for session in sessions:
                session.advance(quantum=1, expansions=2)
        self.assertEqual(sum(s.nodes for s in sessions), 2)
        self.assertEqual(ctx.expanded, 2)
        ctx.deadline = time.perf_counter() - 1
        with self.assertRaisesRegex(BudgetExpired, "timeout"):
            consume(
                ctx.push_policy.distribution(
                    ctx, ctx.start, consume(ctx.children(ctx.start))
                )
            )

    def test_feature_export_matches_runtime_and_cache_is_invocation_local(self):
        row = {
            "id": "test",
            "date": "2026-10-25",
            "name": "tiny",
            "width": 4,
            "height": 1,
            "puzzle": "PO-O",
            "board_sha256": "fixture",
        }
        decisions, _ = export_board((row, "R", "train"))
        ctx = self.context(board_for(row))
        children = consume(ctx.children(ctx.start))
        runtime = consume(action_features(ctx, ctx.start, children))
        self.assertEqual(decisions[0][0], runtime)
        self.assertEqual(len(runtime[0]), len(FEATURES))
        self.assertTrue(all(math.isfinite(v) for xs in runtime for v in xs))
        first = consume(ctx.push_policy.distribution(ctx, ctx.start, children))
        self.assertEqual(
            consume(ctx.push_policy.distribution(ctx, ctx.start, children)), first
        )
        self.assertEqual(ctx.metrics["policy_cache_hits"], 1)
        other = self.context(board_for(row))
        consume(
            other.push_policy.distribution(
                other, other.start, consume(other.children(other.start))
            )
        )
        self.assertEqual(other.metrics["policy_cache_hits"], 0)
        self.assertIsNot(other.policy_cache, ctx.policy_cache)
        with self.assertRaisesRegex(ValueError, "primitive"):
            consume(
                action_features(
                    ctx,
                    ctx.start,
                    [replace(children[0], pushes=children[0].pushes * 2)],
                )
            )

    def test_experiment_requires_explicit_model_and_preserves_baseline_routing(self):
        board = [["U", "O", "", "O"]]
        for kwargs in (
            {"strategy": "phs"},
            {"schedule": "phs_beam"},
            {"policy_path": self.path},
        ):
            with self.assertRaisesRegex(ValueError, "explicit policy"):
                solve_v4(board, **kwargs)
        result = solve_v4(
            board, schedule="phs_beam", policy_path=self.path, quality_timeout_seconds=0
        )
        self.assertTrue(result.solved)
        self.assertEqual(
            [a.strategy for a in result.attempts],
            ["rank_discrepancy", "weighted", "macro", "committed_beam", "phs"],
        )
        self.assertEqual(
            sum(a.nodes_expanded for a in result.attempts), result.nodes_expanded
        )
        result = solve_v4(
            board,
            schedule="phs_star_beam",
            policy_path=self.path,
            quality_timeout_seconds=0,
        )
        self.assertEqual(result.attempts[-1].strategy, "phs_star")
        result = solve_v4(
            board,
            schedule="phs_adaptive",
            policy_path=self.path,
            quality_timeout_seconds=0,
        )
        self.assertEqual(result.diagnostics["effective_schedule"], "beam8")
        self.assertLessEqual(result.diagnostics["initial_commitment_count"], 16)

    def test_transforms_inherit_original_identity_and_development_never_calibrates(
        self,
    ):
        from solver.push_solver.benchmark_v4 import corpus
        from solver.push_solver.train_push_policy import DEVELOPMENT

        rows = corpus()
        original = next(r for r in rows if r["id"] == "20260802")
        width = original["width"]
        board = [
            original["puzzle"][i : i + width]
            for i in range(0, len(original["puzzle"]), width)
        ]
        rotated = list(zip(*board[::-1]))
        variant = original | {
            "id": "variant",
            "width": original["height"],
            "height": width,
            "puzzle": "".join("".join(r) for r in rotated),
        }
        split = split_identities(rows + [variant])
        self.assertTrue(all(split[key] == "development" for key in DEVELOPMENT))
        self.assertEqual(split["variant"], split[original["id"]])
        self.assertIn("calibration", split.values())
        self.assertIn("evaluation", split.values())

    def test_offline_fit_keeps_weights_separate_from_supervision(self):
        from hashlib import sha256

        dataset = Path(self.temp.name) / "tiny.npz"
        xs = np.zeros((12, len(FEATURES)))
        xs[::2, FEATURES.index("goal")] = 1
        np.savez_compressed(
            dataset,
            features=xs,
            offsets=np.arange(0, 13, 2),
            labels=np.zeros(6, dtype=int),
            boards=np.repeat(np.arange(3), 2),
        )
        manifest = {
            "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
            "features": FEATURES,
            "manifest": [
                {"split": key} for key in ("train", "calibration", "evaluation")
            ],
            "claim": "test fixture",
        }
        dataset.with_suffix(".manifest.json").write_text(json.dumps(manifest))
        fit(dataset, self.path, epochs=10)
        policy = PushPolicy(self.path)
        self.assertGreater(policy.weights[FEATURES.index("goal")], 0)
        self.assertEqual(
            set(policy.payload),
            {"schema", "name", "weights", "temperature", "uniform_mix"},
        )
        report = json.loads(self.path.with_suffix(".training.json").read_text())
        self.assertLess(
            report["metrics"]["evaluation"]["calibrated"]["mean_teacher_nll"],
            math.log(2),
        )
        fit(dataset, self.path, epochs=5, architecture="mlp")
        self.assertIsNotNone(PushPolicy(self.path).network)

    def test_nonlinear_policy_matches_training_math_and_rejects_bad_shapes(self):
        from solver.push_solver.push_policy import MLP_SCHEMA
        from solver.push_solver.train_push_policy import probabilities

        rng = np.random.default_rng(4)
        model = {
            "hidden_weights": rng.normal(size=(len(FEATURES), 3)),
            "hidden_bias": rng.normal(size=3),
            "output_weights": rng.normal(size=3),
        }
        self.payload = {
            "schema": MLP_SCHEMA,
            "name": "test",
            **{k: v.tolist() for k, v in model.items()},
            "temperature": 1.5,
            "uniform_mix": 0.05,
        }
        policy = self.save()
        xs = rng.normal(size=(4, len(FEATURES)))
        expected, _ = probabilities(
            xs[None],
            np.ones((1, 4), dtype=bool),
            model,
            policy.temperature,
            policy.uniform_mix,
        )
        self.assertTrue(
            np.allclose(policy.probabilities(xs.tolist()), expected[0], atol=1e-14)
        )
        self.payload["hidden_weights"] = [[1.0]]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.save()

    def test_policy_search_matches_independent_small_board_oracle(self):
        import random
        from collections import deque

        from solver.legacy_solver import apply_single_move, lost_check, solved

        rng = random.Random(26)
        for case in range(20):
            layout = (
                ["U", "O", "O", "O", "X", "B", "", "", ""]
                if case % 2
                else ["U", "O", "O", "X", "X", "", "", "", ""]
            )
            rng.shuffle(layout)
            board = tuple(tuple(layout[i : i + 3]) for i in range(0, 9, 3))
            pending, seen, possible = deque([board]), {board}, False
            while pending:
                state = pending.popleft()
                if lost_check(state):
                    continue
                if solved(state):
                    possible = True
                    break
                for move in "UDLR":
                    child = apply_single_move(state, move)
                    if child not in seen:
                        seen.add(child)
                        pending.append(child)
            for kind in ("phs", "levin", "phs_star"):
                result = solve_v4(
                    board,
                    strategy=kind,
                    policy_path=self.path,
                    timeout_seconds=1,
                    quality_timeout_seconds=0,
                )
                self.assertEqual(result.solved, possible, (kind, board))
                if result.solved:
                    self.assertTrue(verify_solution(board, result.moves).ok)

    def test_benchmark_resume_rejects_a_changed_policy(self):
        import subprocess
        import sys

        destination = Path(self.temp.name) / "bench.jsonl"
        command = [
            sys.executable,
            "-m",
            "solver.push_solver.benchmark_v4",
            "--ids",
            "20250101",
            "--strategy",
            "phs",
            "--policy",
            str(self.path),
            "--output",
            str(destination),
            "--seconds",
            "0.2",
        ]
        first = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.payload["weights"] = {FEATURES[0]: 1.0}
        self.save()
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("configuration changed", second.stderr)
        self.assertEqual(len(destination.read_text().splitlines()), 1)
