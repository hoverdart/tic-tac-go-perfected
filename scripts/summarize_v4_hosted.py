"""Independently replay complete hosted release cohorts and summarize moves."""

import argparse
import json
from datetime import date
from pathlib import Path
from statistics import mean, median

from apps.api.puzzle_catalog import CatalogEntry
from scripts.verify_v4_release import GUARDS
from solver.push_solver import verify_solution
from solver.push_solver.benchmark_v4 import (
    ENGINE_SHA256,
    RUNNER_SHA256,
    TARGETS,
    corpus,
    gate,
    read_rows,
    repetition_gate,
)
from solver.push_solver.models import DIRECTION_BY_MOVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    catalog = corpus()
    entries = {row["id"]: row for row in catalog}
    metadata = json.loads((args.directory / "corpus.metadata.json").read_text())
    native = metadata["config"]["provenance"]["runtime"]["native_build"]
    revision = metadata["revision"]
    cohorts = {}
    summary = {"engine_sha256": ENGINE_SHA256, "runner_sha256": RUNNER_SHA256,
               "revision": revision, "native_build": native, "cohorts": {}}
    for name, expected in (("smoke", 14), ("guards", 30), ("matrix", 180), ("corpus", 397)):
        rows = read_rows(args.directory / (name + ".jsonl"))
        assert len(rows) == expected
        assert json.loads((args.directory / (name + ".gate.json")).read_text())["passed"]
        for row in rows:
            entry = entries[row["id"]]
            board = CatalogEntry(date.fromisoformat(entry["date"]), entry["name"],
                                 entry["width"], entry["height"], entry["puzzle"]).board()
            result = row["result"]
            assert row["engine_sha256"] == ENGINE_SHA256 and row["runner_sha256"] == RUNNER_SHA256
            assert row["revision"] == revision and row["board_sha256"] == entry["board_sha256"]
            assert row["routes"] == 2 and row["schedule"] == "coverage" and row["quality_timeout_seconds"] == 30
            assert row["wall_seconds"] <= 30 and result["nodes_expanded"] <= 500000
            assert row["verified"] and row["independently_verified"] and result["solved"]
            assert result["diagnostics"]["exact_hints"] is False
            assert result["diagnostics"]["runtime"]["implementation"] == "cython"
            assert result["diagnostics"]["runtime"]["native_build"] == native
            if row["process"] == "fresh":
                assert 0 < row["process_wall_seconds"] <= 30
            replay = verify_solution(board, result["moves"])
            assert replay.ok and replay.final_board == tuple(tuple(line) for line in result["final_board"])
            assert len(result["moves"]) <= result["baseline_keystrokes"]
            counters = result["diagnostics"]["counters"]
            assert result["quality_nodes_expanded"] == counters.get("vicinity_expansions", 0) + counters.get("keystroke_search_expansions", 0)
            player = next((r, c) for r, line in enumerate(board) for c, cell in enumerate(line) if cell == "U")
            pushes = []
            for move in result["moves"]:
                dr, dc = DIRECTION_BY_MOVE[move]
                r, c = player
                nr, nc = r + dr, c + dc
                piece = board[nr][nc]
                if piece in ("O", "X"):
                    pushes.append({"piece": piece, "cell": nr * len(board[0]) + nc, "move": move})
                    board[nr + dr][nc + dc] = piece
                board[r][c], board[nr][nc] = "", "U"
                player = nr, nc
            assert pushes == result["pushes"]
        cohorts[name] = rows
        baseline = sum(row["result"]["baseline_keystrokes"] for row in rows)
        final = sum(len(row["result"]["moves"]) for row in rows)
        summary["cohorts"][name] = {
            "receipts": len(rows), "independently_replayed_paths_and_push_traces": len(rows),
            "mean_moves": mean(len(row["result"]["moves"]) for row in rows),
            "median_moves": median(len(row["result"]["moves"]) for row in rows),
            "first_incumbent_mean_moves": baseline / len(rows),
            "quality_reduction_percent": 100 * (1 - final / baseline),
            "max_seconds": max(row["wall_seconds"] for row in rows),
            "max_process_seconds": max(row.get("process_wall_seconds", 0) for row in rows),
            "peak_process_rss_bytes": max(row["result"]["diagnostics"]["runtime"]["peak_rss_bytes"] for row in rows),
            "optimizer_errors": sum(row["result"]["diagnostics"]["counters"].get("optimizer_errors", 0) for row in rows),
        }
    assert repetition_gate(cohorts["matrix"])["passed"]
    expected_guards = {(key, process, repeat) for key in GUARDS
                       for process in ("fresh", "warm") for repeat in range(3)}
    assert {(row["id"], row["process"], row["repeat"]) for row in cohorts["guards"]} == expected_guards
    assert {row["id"] for row in cohorts["smoke"]} == set(TARGETS) | set(GUARDS)
    root = Path(__file__).resolve().parents[1]
    coverage = gate(cohorts["corpus"], read_rows(root / "debug-artifacts/v4-2026-10-25/v3-baseline.jsonl"), catalog)
    assert coverage["passed"] and coverage["v4_verified"] == 397
    summary["coverage_gate"] = coverage
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
