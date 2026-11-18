"""Iterative repairs retain valid incumbents and never manufacture proofs."""
import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from unittest.mock import patch

from apps.api.puzzle_catalog import CatalogEntry, CatalogError
from scripts import update_optimized_v1 as worker
from scripts.update_optimized_v1 import REPAIR_RADII, first_win, solve_entry


@dataclass
class Seed:
    moves: str


@dataclass
class Result:
    moves: str
    lower_bound: int = 0
    optimality_proven: bool = False

    def to_dict(self):
        return vars(self)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.entry = CatalogEntry(date(2026, 11, 11), "Fixture", 4, 3, "P-----OO----")

    def test_catalog_failure_retains_diagnostics_without_changing_feed_or_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            feed = state / "optimized-v1.json"
            worker.save_feed(feed, {"2026-11-11": worker.make_record(self.entry, "RD")})
            before = feed.read_bytes()
            with (
                patch("sys.argv", ["update_optimized_v1.py", "--state-dir", directory,
                                   "--through", "2026-11-18", "--publish"]),
                patch.object(worker, "publication_checkout", return_value=state / "publication"),
                patch.object(worker, "fetch_catalog", side_effect=CatalogError("incomplete catalog")),
                patch.object(worker, "publish") as publish,
                patch.object(worker.subprocess, "run") as run,
                self.assertRaisesRegex(CatalogError, "incomplete catalog"),
            ):
                worker.main()
            diagnostic = json.loads((state / "last-run.json").read_text())
            self.assertEqual(diagnostic["boards"], 1)
            self.assertEqual(diagnostic["failures"], [
                {"stage": "catalog", "error": "incomplete catalog"}
            ])
            self.assertIn("finished_at", diagnostic)
            self.assertEqual(feed.read_bytes(), before)
            publish.assert_not_called()
            run.assert_not_called()

    def test_local_catalog_errors_replace_stale_diagnostics_and_preserve_feed(self):
        row = {
            "date": self.entry.puzzle_date.isoformat(), "name": self.entry.title,
            "width": self.entry.width, "height": self.entry.height,
            "puzzle": self.entry.level, "board_sha256": "wrong hash",
        }
        for content in [None, "not json", "{}", json.dumps(row)]:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                source = state / "catalog.jsonl"
                if content is not None:
                    source.write_text(content)
                feed = state / "optimized-v1.json"
                worker.save_feed(feed, {"2026-11-11": worker.make_record(self.entry, "RD")})
                before = feed.read_bytes()
                diagnostic = state / "last-run.json"
                diagnostic.write_text('{"finished_at":"stale","failures":[]}')
                with (
                    patch("sys.argv", ["update_optimized_v1.py", "--state-dir", directory,
                                       "--through", "2026-11-18", "--catalog", str(source)]),
                    self.assertRaisesRegex(CatalogError, "Could not load local board catalog"),
                ):
                    worker.main()
                report = json.loads(diagnostic.read_text())
                self.assertEqual(report["boards"], 1)
                self.assertNotEqual(report["finished_at"], "stale")
                self.assertEqual(report["failures"][0]["stage"], "catalog")
                self.assertEqual(feed.read_bytes(), before)

    def test_repairs_reseed_incumbent_and_keep_exact_bound_separate(self):
        results = [Result("RD"), *[Result("DDRU") for _ in REPAIR_RADII[1:]], Result("DDRU", 1)]
        with patch("solver.push_solver.v4.solve_v4", return_value=Seed("DDRU")), patch(
            "scripts.update_optimized_v1.optimize", side_effect=results
        ) as optimize:
            record, receipt = solve_entry(self.entry, 0.1, 10)
        self.assertEqual(record["moves"], "RD")
        self.assertFalse(record["optimality_proven"])
        self.assertEqual(record["lower_bound"], 1)
        self.assertEqual(len(receipt["repairs"]), len(REPAIR_RADII))
        for call in optimize.call_args_list[1:-1]:
            self.assertEqual(call.args[1], ["RD"])
        self.assertEqual(optimize.call_args.kwargs["mode"], "exact")

    def test_invalid_repair_cannot_enter_incumbent(self):
        with patch("solver.push_solver.v4.solve_v4", return_value=Seed("DDRU")), patch(
            "scripts.update_optimized_v1.optimize", return_value=Result("L")
        ), self.assertRaises(ValueError):
            solve_entry(self.entry, 0.1, 10)

    def test_search_party_95_moves_replays_to_first_win(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/search-party-2026-11-14.json").read_text())
        self.assertEqual(len(fixture["moves"]), 95)
        self.assertEqual(first_win(fixture["board"], fixture["moves"]), fixture["moves"])


if __name__ == "__main__":
    unittest.main()
