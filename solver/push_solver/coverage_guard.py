"""Fail a release when a push-solver benchmark loses solved boards.

Example:

    python3 -m solver.push_solver.coverage_guard \
      debug-artifacts/push_vs_beam_full.csv \
      /tmp/push_solver_v3.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes"}


def solved_ids(path: Path, *, verified_column: str = "push_verified") -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and verified_column not in rows[0]:
        raise ValueError(f"{path} has no {verified_column!r} column")
    return {
        str(row.get("board_id", ""))
        for row in rows
        if str(row.get(verified_column, "")).strip().lower() in TRUE_VALUES
    }


def coverage_regressions(
    baseline: Path,
    candidate: Path,
    *,
    baseline_column: str = "push_verified",
    candidate_column: str = "push_verified",
) -> tuple[set[str], set[str], set[str]]:
    baseline_solved = solved_ids(baseline, verified_column=baseline_column)
    candidate_solved = solved_ids(candidate, verified_column=candidate_column)
    return (
        baseline_solved,
        candidate_solved,
        baseline_solved - candidate_solved,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-column", default="push_verified")
    parser.add_argument("--candidate-column", default="push_verified")
    args = parser.parse_args()

    baseline, candidate, regressions = coverage_regressions(
        args.baseline,
        args.candidate,
        baseline_column=args.baseline_column,
        candidate_column=args.candidate_column,
    )
    print(
        f"coverage baseline={len(baseline)} candidate={len(candidate)} "
        f"regressions={len(regressions)}"
    )
    if regressions:
        print("lost_board_ids=" + ",".join(sorted(regressions)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
