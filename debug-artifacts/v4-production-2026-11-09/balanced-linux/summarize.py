"""Replay all compiled Linux cohorts and rebuild strict release summaries."""

import json
from datetime import date
from pathlib import Path
from statistics import mean, median

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver import verify_solution
from solver.push_solver.benchmark_v4 import (
    ENGINE_SHA256, RUNNER_SHA256, corpus, gate, read_rows, repetition_gate,
)
from solver.push_solver.models import DIRECTION_BY_MOVE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
catalog = corpus()
entries = {row["id"]: row for row in catalog}
files = {"corpus": 397, "fresh": 90, "warm": 90,
         "guards-fresh": 15, "guards-warm": 15, "smoke": 14}
image = json.loads((HERE / "image-runtime.json").read_text())
summary = {"engine_sha256": ENGINE_SHA256, "runner_sha256": RUNNER_SHA256,
           "cohorts": {}}
cohorts = {}
for name, count in files.items():
    rows = read_rows(HERE / (name + ".jsonl"))
    assert len(rows) == count, (name, len(rows), count)
    for row in rows:
        entry = entries[row["id"]]
        board = CatalogEntry(date.fromisoformat(entry["date"]), entry["name"],
                             entry["width"], entry["height"], entry["puzzle"]).board()
        result = row["result"]
        assert row["engine_sha256"] == ENGINE_SHA256 and row["runner_sha256"] == RUNNER_SHA256
        assert row["board_sha256"] == entry["board_sha256"]
        assert row["routes"] == 2 and result["diagnostics"]["route_limit"] == 2
        replay = verify_solution(board, result["moves"])
        assert replay.ok and row["verified"] and result["solved"]
        assert replay.final_board == tuple(tuple(line) for line in result["final_board"])
        assert row["wall_seconds"] <= 30 and result["nodes_expanded"] <= 500000
        assert len(result["moves"]) <= result["baseline_keystrokes"]
        assert result["diagnostics"]["exact_hints"] is False
        runtime = result["diagnostics"]["runtime"]
        assert runtime["implementation"] == "cython"
        assert runtime["native_build"] == image["runtime"]["native_build"]
        assert runtime["peak_rss_bytes"] < 4 * 1024 ** 3
        if row["process"] == "fresh":
            assert 0 < row["process_wall_seconds"] <= 30
        player = next((r, c) for r, line in enumerate(board)
                      for c, cell in enumerate(line) if cell == "U")
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
    summary["cohorts"][name] = {
        "receipts": len(rows), "independently_replayed_paths_and_push_traces": len(rows),
        "mean_moves": mean(len(r["result"]["moves"]) for r in rows),
        "median_moves": median(len(r["result"]["moves"]) for r in rows),
        "max_seconds": max(r["wall_seconds"] for r in rows),
        "max_process_seconds": max(r.get("process_wall_seconds", 0) for r in rows),
        "peak_process_rss_bytes": max(r["result"]["diagnostics"]["runtime"]["peak_rss_bytes"] for r in rows),
    }
matrix = repetition_gate(cohorts["fresh"] + cohorts["warm"])
reference = read_rows(ROOT / "debug-artifacts/v4-2026-10-25/v3-baseline.jsonl")
coverage = gate(cohorts["corpus"], reference, catalog)
assert matrix["passed"] and coverage["passed"] and coverage["v4_verified"] == 397
summary.update(repetition_gate=matrix, coverage_gate=coverage)
(HERE / "repetitions.gate.json").write_text(json.dumps(matrix, indent=2) + "\n")
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
