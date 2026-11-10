import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from scripts.select_daily_solver import validate_evidence


class ReleaseSelectionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "debug-artifacts/qualification"
        self.directory.mkdir(parents=True)
        self.revision = "tic-tac-go-00058-test"
        sources = {}
        for name in ("v4.py", "path_optimizer.py", "benchmark_v4.py"):
            path = self.root / "solver/push_solver" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            sources[str(path.relative_to(self.root))] = sha256(path.read_bytes()).hexdigest()
        native = {"cython": "3.3.1", "extensions": {"v4.so": "one", "path_optimizer.so": "two"}}
        metadata = {
            "revision": self.revision, "routes": 2, "engine_sha256": "engine", "runner_sha256": "runner",
            "config": {"provenance": {"sources": sources, "runtime": {"implementation": "cython", "native_build": native}}},
        }
        row = {
            "version": "v4", "verified": True, "independently_verified": True,
            "revision": self.revision, "engine_sha256": "engine", "runner_sha256": "runner",
            "limits": {"seconds": 30, "nodes": 500000}, "schedule": "coverage", "routes": 2,
            "quality_timeout_seconds": 30, "wall_seconds": 1, "process": "fresh", "process_wall_seconds": 1.2,
            "result": {"solved": True, "nodes_expanded": 1, "diagnostics": {
                "exact_hints": False, "route_limit": 2, "runtime": {"implementation": "cython", "native_build": native},
            }},
        }
        for name, count in (("smoke", 12), ("guards", 18), ("matrix", 180), ("corpus", 397)):
            (self.directory / (name + ".gate.json")).write_text(json.dumps({"passed": True, "v4_verified": 397}))
            (self.directory / (name + ".metadata.json")).write_text(json.dumps(metadata))
            (self.directory / (name + ".jsonl")).write_text((json.dumps(row) + "\n") * count)

    def check(self):
        return validate_evidence(self.directory, self.revision, root=self.root)

    def test_complete_matching_qualification_is_accepted(self):
        self.assertEqual(self.check(), "engine")

    def test_failed_gate_and_wrong_revision_are_rejected(self):
        path = self.directory / "smoke.gate.json"
        path.write_text('{"passed": false}')
        with self.assertRaises(ValueError):
            self.check()
        path.write_text('{"passed": true}')
        with self.assertRaises(ValueError):
            validate_evidence(self.directory, "tic-tac-go-other", root=self.root)

    def test_missing_receipt_is_rejected(self):
        path = self.directory / "matrix.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_checkout_source_is_rejected(self):
        (self.root / "solver/push_solver/v4.py").write_text("unqualified change")
        with self.assertRaises(ValueError):
            self.check()

    def test_mismatched_worker_binary_is_rejected(self):
        path = self.directory / "matrix.jsonl"
        rows = path.read_text().splitlines()
        row = json.loads(rows[-1])
        row["result"]["diagnostics"]["runtime"]["native_build"]["extensions"]["v4.so"] = "different"
        rows[-1] = json.dumps(row)
        path.write_text("\n".join(rows) + "\n")
        with self.assertRaises(ValueError):
            self.check()

    def test_evidence_outside_repository_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_evidence(self.root.parent, self.revision, root=self.root)
