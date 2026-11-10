"""Independently replay the complete local production qualification cohorts.

Run from the repository root with PYTHONPATH=. .venv/bin/python <this file>.
"""

import json
from datetime import date
from pathlib import Path
from statistics import mean, median

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver import verify_solution
from solver.push_solver.benchmark_v4 import (
    ENGINE_SHA256,
    RUNNER_SHA256,
    corpus,
    read_rows,
    repetition_gate,
)
from solver.push_solver.models import DIRECTION_BY_MOVE

HERE = Path(__file__).resolve().parent
ENTRIES = {row["id"]: row for row in corpus()}
COHORTS = {"linux-corpus.jsonl": 397, "linux-fresh.jsonl": 90,
           "linux-warm.jsonl": 90, "linux-guards-fresh.jsonl": 6,
           "linux-guards-warm.jsonl": 6}
summary = {"engine_sha256": ENGINE_SHA256, "runner_sha256": RUNNER_SHA256,
           "cohorts": {}}
all_rows = {}
for filename, expected in COHORTS.items():
    rows = read_rows(HERE / filename)
    assert len(rows) == expected, (filename, len(rows), expected)
    for row in rows:
        entry = ENTRIES[row["id"]]
        board = CatalogEntry(date.fromisoformat(entry["date"]), entry["name"],
                             entry["width"], entry["height"], entry["puzzle"]).board()
        result = row["result"]
        replay = verify_solution(board, result["moves"])
        assert replay.ok and row["verified"] and result["solved"]
        assert row["engine_sha256"] == ENGINE_SHA256
        assert row["board_sha256"] == entry["board_sha256"]
        assert replay.final_board == tuple(tuple(line) for line in result["final_board"])
        assert row["wall_seconds"] <= 30 and result["nodes_expanded"] <= 500000
        assert len(result["moves"]) <= result["baseline_keystrokes"]
        if row["process"] == "fresh":
            assert 0 < row["process_wall_seconds"] <= 30
        if filename != "linux-corpus.jsonl":
            assert row["runner_sha256"] == RUNNER_SHA256
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
    all_rows[filename] = rows
    summary["cohorts"][filename] = {
        "receipts": len(rows), "independently_replayed_paths_and_push_traces": len(rows),
        "mean_moves": mean(len(r["result"]["moves"]) for r in rows),
        "median_moves": median(len(r["result"]["moves"]) for r in rows),
        "max_seconds": max(r["wall_seconds"] for r in rows),
        "max_process_seconds": max(r.get("process_wall_seconds", 0) for r in rows),
    }
matrix = repetition_gate(all_rows["linux-fresh.jsonl"] + all_rows["linux-warm.jsonl"])
assert matrix["passed"], matrix
summary["repetition_gate"] = matrix
(HERE / "linux-repetitions.gate.json").write_text(json.dumps(matrix, indent=2) + "\n")
(HERE / "local-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
