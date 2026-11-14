"""Iterative repairs retain valid incumbents and never manufacture proofs."""
import json
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from unittest.mock import patch

from apps.api.puzzle_catalog import CatalogEntry
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
