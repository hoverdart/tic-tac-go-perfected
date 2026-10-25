from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from solver.push_solver import solve_v4
from solver.push_solver.benchmark_v4 import (
    corpus,
    gate,
    repetition_gate,
    TARGETS,
    ENGINE_SHA256,
)
from solver.push_solver.train_v4_candidate import identity, splits


class V4ReleaseToolsTest(unittest.TestCase):
    def test_corpus_hashes_and_release_gates_fail_closed(self):
        boards = corpus()
        self.assertEqual(len(boards), 397)
        rows = [
            {
                "id": r["id"],
                "board_sha256": r["board_sha256"],
                "version": "v4",
                "engine_sha256": ENGINE_SHA256,
                "limits": {"seconds": 30, "nodes": 500000},
                "process_wall_seconds": 1.0,
                "verified": True,
                "wall_seconds": 1.0,
                "result": {
                    "solved": True,
                    "nodes_expanded": 1,
                    "diagnostics": {"exact_hints": False},
                },
            }
            for r in boards
        ]
        baseline = [{**r, "version": "v3"} for r in rows]
        self.assertTrue(gate(rows, baseline, boards)["passed"])
        self.assertFalse(gate(rows + [rows[0]], baseline, boards)["passed"])
        for invalid in (
            "missing",
            "hash",
            "timeout",
            "hints",
            "nodes",
            "engine",
            "budget",
            "strategy",
        ):
            altered = deepcopy(rows)
            if invalid == "missing":
                altered.pop()
            if invalid == "hash":
                altered[-1]["board_sha256"] = "wrong"
            if invalid == "timeout":
                altered[-1]["wall_seconds"] = 30.1
            if invalid == "hints":
                altered[-1]["result"]["diagnostics"]["exact_hints"] = True
            if invalid == "nodes":
                altered[-1]["result"]["nodes_expanded"] = 500001
            if invalid == "engine":
                altered[-1]["engine_sha256"] = "stale"
            if invalid == "budget":
                altered[-1]["limits"]["seconds"] = 300
            if invalid == "strategy":
                altered[-1]["strategy_requested"] = "rank_discrepancy"
            self.assertFalse(gate(altered, baseline, boards)["passed"], invalid)
        self.assertFalse(repetition_gate(rows)["passed"])
        matrix = [
            {
                **next(r for r in rows if r["id"] == key),
                "repeat": repeat,
                "process": process,
            }
            for key in TARGETS
            for process in ("fresh", "warm")
            for repeat in range(10)
        ]
        self.assertTrue(repetition_gate(matrix)["passed"])
        matrix[-1]["verified"] = False
        self.assertFalse(repetition_gate(matrix)["passed"])

    def test_transforms_and_duplicates_share_split_and_targets_are_development(self):
        rows = corpus()
        split = splits(rows)
        self.assertTrue(all(split[key] == "development" for key in TARGETS))
        original = rows[1]
        width = original["width"]
        board = [
            original["puzzle"][r * width : (r + 1) * width]
            for r in range(original["height"])
        ]
        rotated = list(zip(*board[::-1]))
        variant = {
            **original,
            "id": "transformed",
            "width": original["height"],
            "height": width,
            "puzzle": "".join("".join(r) for r in rotated),
        }
        self.assertEqual(identity(original), identity(variant))
        split = splits(rows + [variant])
        self.assertEqual(split[original["id"]], split["transformed"])

    def test_candidate_model_cannot_enable_exact_action_hints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(
                json.dumps({"weights": {}, "state_action_hints": {"remembered": 100}})
            )
            with self.assertRaisesRegex(ValueError, "without exact action hints"):
                solve_v4([["U", "O", "O"]], ranker_path=path, quality_timeout_seconds=0)
            path.write_text(json.dumps({"weights": {}, "state_action_hints": {}}))
            result = solve_v4(
                [["U", "O", "", "O"]], ranker_path=path, quality_timeout_seconds=0
            )
            self.assertTrue(result.solved)
            self.assertTrue(result.diagnostics["model_hashes"]["candidate"])
            self.assertFalse(result.diagnostics["exact_hints"])


if __name__ == "__main__":
    unittest.main()
