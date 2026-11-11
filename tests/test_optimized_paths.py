"""Compare move costs and certificates with independent unit-step BFS."""

import json
import random
import subprocess
import sys
import time
import unittest
from collections import deque
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from solver.board_utils import normalize_board
from solver.legacy_solver import apply_single_move, lost_check, solved
from solver.optimized_paths import optimize
from solver.optimized_paths.bank import load_bank
from solver.optimized_paths.benchmark import configuration_sha256
from solver.optimized_paths.lower_bound import build_pattern_database
from solver.optimized_paths.model import Budget, Geometry, LimitReached
from solver.optimized_paths.search import bounded_astar
from solver.optimized_paths.verify_benchmark import verify_cohort
from solver.push_solver.verify import verify_solution


def oracle(board):
    start = normalize_board(board)
    if lost_check(start):
        return None
    queue = deque([(start, "")])
    seen = {start}
    while queue:
        current, moves = queue.popleft()
        if solved(current):
            return moves
        for direction in "UDLR":
            child = apply_single_move(current, direction)
            if child not in seen and not lost_check(child):
                seen.add(child)
                queue.append((child, moves + direction))
    return None


def budget(nodes=100_000):
    return Budget(time.perf_counter() + 10, nodes)


REPAIR_BOARD = [
    ["X", "B", "U", ""],
    ["", "", "O", ""],
    ["", "O", "", ""],
    ["X", "", "", ""],
]
REPAIR_PATH = "RDDLULLDRUR"


class OptimizedPathsTest(unittest.TestCase):
    def test_graph_combines_exact_state_trajectory_segments(self):
        board = [[""] * 5 for _ in range(5)]
        board[4][0], board[1][1], board[1][3] = "U", "O", "O"
        seeds = ["URRRULU", "UURDRUU"]
        self.assertTrue(all(verify_solution(board, path).ok for path in seeds))
        result = optimize(board, seeds, mode="graph", seconds=1)
        self.assertEqual(result.baseline_moves, 7)
        self.assertEqual(len(result.moves), 5)
        self.assertTrue(result.improved)
        self.assertFalse(result.optimality_proven)
        self.assertEqual(result.lower_bound, 0)

    def test_exact_search_changes_push_direction_and_proves_shortest_path(self):
        result = optimize(REPAIR_BOARD, [REPAIR_PATH], mode="exact", seconds=2)
        self.assertEqual(len(oracle(REPAIR_BOARD)), 3)
        self.assertEqual(len(result.moves), 3)
        self.assertEqual(result.lower_bound, 3)
        self.assertTrue(result.optimality_proven)
        self.assertTrue(verify_solution(REPAIR_BOARD, result.moves).ok)

    def test_vicinity_improves_without_claiming_global_optimality(self):
        result = optimize(REPAIR_BOARD, [REPAIR_PATH], mode="repair", seconds=2)
        self.assertEqual(len(result.moves), 3)
        self.assertFalse(result.optimality_proven)
        self.assertEqual(result.lower_bound, 0)

    def test_timeout_keeps_the_verified_seed_and_never_claims_a_proof(self):
        result = optimize(REPAIR_BOARD, [REPAIR_PATH], mode="exact", seconds=0)
        self.assertEqual(result.moves, REPAIR_PATH)
        self.assertEqual(result.termination, "timeout")
        self.assertEqual(result.expanded, 0)
        self.assertFalse(result.optimality_proven)

    def test_shared_node_cap_is_not_exceeded(self):
        for mode in ("graph", "repair", "exact"):
            result = optimize(
                REPAIR_BOARD, [REPAIR_PATH], mode=mode, seconds=1, max_nodes=1
            )
            self.assertLessEqual(result.expanded, 1)
            self.assertEqual(result.moves, REPAIR_PATH)
            self.assertTrue(result.verified)
            self.assertFalse(result.optimality_proven)

    def test_invalid_seed_is_rejected_even_with_zero_time(self):
        for seeds in ([], ["U"], [REPAIR_PATH, "U"], [None], "R"):
            with self.subTest(seeds=seeds), self.assertRaises((ValueError, TypeError)):
                optimize(REPAIR_BOARD, seeds, seconds=0)

    def test_initial_ooo_win_is_trivially_optimal_with_no_budget(self):
        result = optimize([["U", "", "O", "O", "O"]], ["R"], mode="exact", seconds=1)
        self.assertEqual(result.moves, "")
        self.assertTrue(result.optimality_proven)
        zero = optimize([["U", "", "O", "O", "O"]], [""], seconds=0)
        self.assertTrue(zero.optimality_proven)

    def test_nontrivial_ooo_finish_is_supported_without_two_o_database(self):
        board = [["U", "O", "", "O", "O"]]
        result = optimize(board, ["R"], mode="exact", seconds=1)
        self.assertEqual(result.moves, "R")
        self.assertEqual(result.lower_bound, 1)
        self.assertTrue(result.optimality_proven)
        self.assertEqual(result.phases[-2]["termination"], "additional_o_fallback")

    def test_final_walking_cost_uses_the_nearest_winning_cell(self):
        board = [[""] * 5 for _ in range(3)]
        board[0][4], board[1][1], board[1][2] = "U", "O", "O"
        result = optimize(board, ["LLLLD"], mode="exact", seconds=2)
        self.assertEqual(len(result.moves), 2)
        self.assertEqual(result.lower_bound, 2)
        self.assertTrue(result.optimality_proven)

    def test_restricted_anchor_cap_preserves_incumbent(self):
        result = optimize(
            REPAIR_BOARD, [REPAIR_PATH], mode="repair", seconds=1, max_layouts=0
        )
        self.assertTrue(result.verified)
        self.assertLessEqual(len(result.moves), len(REPAIR_PATH))
        self.assertEqual(result.termination, "anchor_layout_cap")
        self.assertFalse(result.optimality_proven)

    def test_partial_expansion_remains_in_global_lower_bound(self):
        geometry = Geometry(REPAIR_BOARD)

        def interrupted(_self, _state, _walks):
            raise LimitReached("timeout")
            yield  # Make the interruption occur within successor generation.

        with patch.object(Geometry, "successors", interrupted):
            moves, lower, reason, _states = bounded_astar(
                geometry, REPAIR_PATH, budget()
            )
        self.assertEqual(moves, REPAIR_PATH)
        self.assertEqual(lower, 0)
        self.assertEqual(reason, "timeout")

    def test_node_cap_after_pop_keeps_active_label_in_bound(self):
        moves, lower, reason, _states = bounded_astar(
            Geometry(REPAIR_BOARD), REPAIR_PATH, budget(0)
        )
        self.assertEqual(moves, REPAIR_PATH)
        self.assertEqual(lower, 0)
        self.assertEqual(reason, "node_cap")

    def test_invalid_configuration(self):
        for options in (
            {"mode": "other"},
            {"seconds": float("nan")},
            {"seconds": -1},
            {"max_nodes": -1},
            {"radius": 1.5},
            {"max_layouts": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                optimize(REPAIR_BOARD, [REPAIR_PATH], **options)

    def test_random_small_exact_results_match_independent_unit_step_bfs(self):
        randomizer = random.Random(728)
        checked = 0
        for _ in range(45):
            board = [[""] * 3 for _ in range(3)]
            chosen = randomizer.sample(range(9), 5)
            for cell, value in zip(chosen, ["U", "O", "O", "X", "B"]):
                board[cell // 3][cell % 3] = value
            answer = oracle(board)
            if answer is None:
                continue
            result = optimize(board, [answer], mode="exact", seconds=2)
            self.assertEqual(len(result.moves), len(answer))
            self.assertEqual(result.lower_bound, len(answer))
            self.assertTrue(result.optimality_proven)
            checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_single_case_cli_emits_verified_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"board": REPAIR_BOARD, "seeds": [REPAIR_PATH]}))
            output = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "solver.optimized_paths",
                    "--input",
                    str(path),
                    "--mode",
                    "exact",
                    "--seconds",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            row = json.loads(output.stdout)
            self.assertTrue(row["optimality_proven"])
            self.assertEqual(row["move_count"], 3)


class RelaxedPatternDatabaseTest(unittest.TestCase):
    def test_complete_table_matches_independent_bfs_with_walls(self):
        base = [["U", "", ""], ["", "O", "O"], ["B", "", ""]]
        geometry = Geometry(base)
        table = build_pattern_database(geometry, budget())
        self.assertTrue(table.complete)
        randomizer = random.Random(19)
        floor = [i for i in range(9) if base[i // 3][i % 3] != "B"]
        for _ in range(30):
            oa, ob, player = randomizer.sample(floor, 3)
            board = [["" if value != "B" else "B" for value in row] for row in base]
            for cell, value in ((oa, "O"), (ob, "O"), (player, "U")):
                board[cell // 3][cell % 3] = value
            answer = oracle(board)
            distance = table.estimate(((1 << oa) | (1 << ob), 0, player))
            self.assertEqual(distance, float("inf") if answer is None else len(answer))

    def test_partial_tables_are_discarded_and_cannot_prune(self):
        geometry = Geometry(REPAIR_BOARD)
        for limit in (0, 1, 10):
            table = build_pattern_database(geometry, budget(), max_states=limit)
            self.assertFalse(table.complete)
            self.assertEqual(table.distances, {})
            self.assertEqual(table.estimate(geometry.start), 0)
        table = build_pattern_database(geometry, Budget(0, 10))
        self.assertFalse(table.complete)
        self.assertEqual(table.reason, "timeout")

    def test_additional_os_use_safe_zero_heuristic(self):
        geometry = Geometry([["U", "", "O", "O", "O"]])
        table = build_pattern_database(geometry, budget())
        self.assertFalse(table.complete)
        self.assertEqual(table.reason, "additional_o_fallback")
        self.assertEqual(table.estimate(geometry.start), 0)

    def test_relaxation_never_exceeds_real_distance_with_xs(self):
        geometry = Geometry(REPAIR_BOARD)
        table = build_pattern_database(geometry, budget())
        self.assertTrue(table.complete)
        self.assertLessEqual(table.estimate(geometry.start), len(oracle(REPAIR_BOARD)))


class GeometryTest(unittest.TestCase):
    def test_steps_match_legacy_replayer_and_never_wrap_rows(self):
        board = [["", "X", "U", ""], ["", "O", "", "O"], ["", "B", "", ""]]
        geometry = Geometry(board)
        queue = deque([(geometry.start, normalize_board(board))])
        seen = {geometry.start}
        for _ in range(150):
            if not queue:
                break
            state, rendered = queue.popleft()
            for move in "UDLR":
                expected = apply_single_move(rendered, move)
                actual = geometry.step(state, move)
                if expected == rendered or lost_check(expected):
                    self.assertIsNone(actual)
                else:
                    self.assertEqual(actual, Geometry(expected).start)
                    if actual not in seen:
                        seen.add(actual)
                        queue.append((actual, expected))
        self.assertGreater(len(seen), 30)

    def test_intermediate_x_loss_rejected(self):
        board = [["X", "X", "", "", ""], ["", "", "X", "", ""], ["", "", "U", "O", "O"]]
        self.assertIsNone(Geometry(board).step(Geometry(board).start, "U"))


class BankTest(unittest.TestCase):
    def inputs(self, root, rows):
        puzzle = "POO------"
        digest = sha256(f"3x3:{puzzle}".encode()).hexdigest()
        catalog = {
            "id": "20250101",
            "date": "2025-01-01",
            "name": "Fixture",
            "width": 3,
            "height": 3,
            "puzzle": puzzle,
            "board_sha256": digest,
        }
        (root / "catalog.jsonl").write_text(json.dumps(catalog) + "\n")
        (root / "paths.jsonl").write_text(
            "".join(
                json.dumps({"id": "20250101", "board_sha256": digest, **r}) + "\n"
                for r in rows
            )
        )
        manifest = root / "inputs.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "optimized-path-inputs-v1",
                    "catalog": "catalog.jsonl",
                    "sources": ["paths.jsonl"],
                }
            )
        )
        return manifest

    def test_empty_paths_are_valid_and_duplicates_keep_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bank = load_bank(
                self.inputs(root, [{"solution": ""}, {"result": {"moves": ""}}]),
                root=root,
            )
            self.assertEqual(list(bank.paths["20250101"]), [""])
            self.assertEqual(len(bank.paths["20250101"][""]), 2)
            self.assertEqual(bank.counts["unique_paths_replayed"], 1)

    def test_mismatches_transforms_and_invalid_paths_are_visible_and_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"solution": ""},
                {"solution": "", "board_sha256": "wrong"},
                {"solution": "", "transformation": "rotate90"},
                {"solution": "U"},
                {"solution": "", "id": "20250102"},
                {"result": {"moves": None}},
            ]
            bank = load_bank(self.inputs(root, rows), root=root)
            self.assertEqual(len(bank.paths["20250101"]), 1)
            self.assertEqual(len(bank.rejected), 4)
            for reason in (
                "board_hash_mismatch",
                "transformed_coordinates",
                "invalid_path",
                "unknown_board",
                "rows_without_path",
            ):
                self.assertEqual(bank.counts[reason], 1)

    def test_corrupt_catalog_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.inputs(root, [{"solution": ""}])
            row = json.loads((root / "catalog.jsonl").read_text())
            row["board_sha256"] = "wrong"
            (root / "catalog.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_bank(manifest, root=root)

    def make_cohort(self, root):
        manifest = self.inputs(root, [{"solution": ""}])
        spec = json.loads(manifest.read_text())
        spec["catalog"] = str(root / spec["catalog"])
        spec["sources"] = [str(root / p) for p in spec["sources"]]
        manifest.write_text(json.dumps(spec))
        output = root / "cohort"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "solver.optimized_paths.benchmark",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--mode",
                "exact",
                "--seconds",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return manifest, output

    def test_benchmark_and_independent_receipt_verifier_roundtrip(self):
        with TemporaryDirectory() as directory:
            manifest, output = self.make_cohort(Path(directory))
            report = verify_cohort(output, manifest)
            self.assertTrue(report["all_verified"])
            self.assertEqual(report["independently_verified"], 1)

    def test_receipt_verifier_rejects_duplicate_cases_and_forged_claims(self):
        with TemporaryDirectory() as directory:
            manifest, output = self.make_cohort(Path(directory))
            path = output / "results.jsonl"
            original = path.read_text()
            path.write_text(original + original)
            with self.assertRaises(ValueError):
                verify_cohort(output, manifest)
            row = json.loads(original)
            row["result"]["optimality_proven"] = False
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                verify_cohort(output, manifest)

    def test_receipt_verifier_rejects_changed_sources(self):
        with TemporaryDirectory() as directory:
            manifest, output = self.make_cohort(Path(directory))
            path = output / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["engine_sources"]["solver/optimized_paths/search.py"] = "wrong"
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                verify_cohort(output, manifest)

    def test_receipt_verifier_rejects_changed_configuration(self):
        with TemporaryDirectory() as directory:
            manifest, output = self.make_cohort(Path(directory))
            path = output / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["options"]["seconds"] = 2
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                verify_cohort(output, manifest)

    def test_receipt_verifier_rejects_row_from_other_configuration(self):
        with TemporaryDirectory() as directory:
            manifest, output = self.make_cohort(Path(directory))
            path = output / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["options"]["seconds"] = 2
            metadata["configuration_sha256"] = configuration_sha256(
                metadata["engine_sha256"],
                metadata["inputs"],
                metadata["options"],
                metadata["runtime"],
            )
            path.write_text(json.dumps(metadata))
            summary_path = output / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["configuration_sha256"] = metadata["configuration_sha256"]
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                verify_cohort(output, manifest)


if __name__ == "__main__":
    unittest.main()
