import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from solver.push_solver.coverage_guard import coverage_regressions


class PushSolverCoverageGuardTest(unittest.TestCase):
    def _write(self, path: Path, solved: dict[str, bool]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("board_id", "push_verified"))
            writer.writeheader()
            for board_id, verified in solved.items():
                writer.writerow({"board_id": board_id, "push_verified": verified})

    def test_accepts_equal_or_increased_coverage(self):
        with TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "baseline.csv"
            candidate = Path(temp_dir) / "candidate.csv"
            self._write(baseline, {"a": True, "b": False})
            self._write(candidate, {"a": True, "b": True})

            _baseline, _candidate, regressions = coverage_regressions(
                baseline,
                candidate,
            )

        self.assertEqual(regressions, set())

    def test_reports_every_solved_to_unsolved_regression(self):
        with TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "baseline.csv"
            candidate = Path(temp_dir) / "candidate.csv"
            self._write(baseline, {"a": True, "b": True})
            self._write(candidate, {"a": False, "b": True})

            _baseline, _candidate, regressions = coverage_regressions(
                baseline,
                candidate,
            )

        self.assertEqual(regressions, {"a"})


if __name__ == "__main__":
    unittest.main()
