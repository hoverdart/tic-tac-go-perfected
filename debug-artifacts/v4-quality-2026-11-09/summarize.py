"""Verify complete cohorts, replay moves/pushes, and rebuild quality comparisons."""

import json
from datetime import date
from pathlib import Path
from statistics import mean, median

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver import verify_solution
from solver.push_solver.benchmark_v4 import corpus, gate, read_rows
from solver.push_solver.models import DIRECTION_BY_MOVE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CATALOG = corpus()
ENTRIES = {row["id"]: row for row in CATALOG}
REFERENCE = read_rows(ROOT / "debug-artifacts/v4-2026-10-25/v3-baseline.jsonl")
RAW = read_rows(ROOT / "debug-artifacts/v4-regressions-2026-11-09/corpus-coverage.jsonl")


def verify_rows(rows):
    assert len(rows) == 397 and len({row["id"] for row in rows}) == 397
    verified = 0
    for row in rows:
        entry = ENTRIES[row["id"]]
        assert row["board_sha256"] == entry["board_sha256"]
        board = CatalogEntry(
            date.fromisoformat(entry["date"]), entry["name"],
            entry["width"], entry["height"], entry["puzzle"],
        ).board()
        result = row["result"]
        replay = verify_solution(board, result["moves"])
        assert replay.ok == row["verified"], row["id"]
        if not replay.ok:
            continue
        assert replay.final_board == tuple(tuple(line) for line in result["final_board"])
        player = next((r, c) for r, line in enumerate(board) for c, value in enumerate(line) if value == "U")
        pushes = []
        for move in result["moves"]:
            dr, dc = DIRECTION_BY_MOVE[move]
            r, c = player
            nr, nc = r + dr, c + dc
            piece = board[nr][nc]
            if piece in ("O", "X"):
                pushes.append({"piece": piece, "cell": nr * len(board[0]) + nc, "move": move})
                board[nr + dr][nc + dc] = piece
            board[r][c] = ""
            board[nr][nc] = "U"
            player = nr, nc
        assert pushes == result["pushes"], row["id"]
        verified += 1
    return verified


def statistics(rows):
    solved = [row for row in rows if row["verified"]]
    lengths = [len(row["result"]["moves"]) for row in solved]
    return {
        "attempted": len(rows), "solved": len(solved),
        "mean_moves_on_solved": mean(lengths), "median_moves_on_solved": median(lengths),
        "total_moves_on_solved": sum(lengths),
        "max_wall_seconds": max(row["wall_seconds"] for row in rows),
        "total_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "failures": [row["id"] for row in rows if not row["verified"]],
    }


def paired(baseline, candidate):
    original = {row["id"]: row for row in baseline if row["verified"]}
    shared = [row for row in candidate if row["verified"] and row["id"] in original]
    before = [len(original[row["id"]]["result"]["moves"]) for row in shared]
    after = [len(row["result"]["moves"]) for row in shared]
    return {
        "boards": len(shared), "before_mean": mean(before), "after_mean": mean(after),
        "before_total": sum(before), "after_total": sum(after),
        "reduction_percent": 100 * (1 - sum(after) / sum(before)),
        "shorter": sum(b > a for b, a in zip(before, after)),
        "equal": sum(b == a for b, a in zip(before, after)),
        "longer": sum(b < a for b, a in zip(before, after)),
    }


files = {
    "basic_quality_10": "corpus-basic-quality.jsonl",
    "quality_10": "corpus-quality-10.jsonl",
    "quality_30": "corpus-quality-30.jsonl",
    "v3_quality_10": "v3-quality-corpus.jsonl",
}
cohorts = {name: read_rows(HERE / file) for name, file in files.items()}
if any(len(rows) != 397 for rows in cohorts.values()):
    raise SystemExit("Wait for all four complete 397-board cohorts.")
summary = {"coverage_control": statistics(RAW), "cohorts": {}}
for name, rows in cohorts.items():
    replayed = verify_rows(rows)
    item = {**statistics(rows), "independently_replayed_paths_and_push_traces": replayed}
    if name in ("quality_10", "quality_30"):
        item["gate"] = gate(rows, REFERENCE, CATALOG)
        item["versus_coverage_control"] = paired(RAW, rows)
        item["versus_v3_same_10s_quality_window"] = paired(cohorts["v3_quality_10"], rows)
        item["v3_comparison_quality_window_matches"] = name == "quality_10"
        item["own_first_incumbent_total"] = sum(row["result"]["baseline_keystrokes"] for row in rows)
        item["quality_improved_boards"] = sum(row["result"]["quality_improved"] for row in rows)
        item["peak_process_rss_bytes"] = max(
            row["result"]["diagnostics"]["runtime"]["peak_rss_bytes"] for row in rows
        )
        for row in rows:
            result = row["result"]
            if not row["verified"]:
                continue
            assert len(result["moves"]) <= result["baseline_keystrokes"]
            metrics = result["diagnostics"]["counters"]
            assert result["quality_nodes_expanded"] == (
                metrics.get("vicinity_expansions", 0) + metrics.get("keystroke_search_expansions", 0)
            )
    summary["cohorts"][name] = item
summary["scope"] = (
    "Sequential native development cohorts, with independent experiments overlapping on the host. "
    "V3 comparisons use the identical solved board identities; V4 full means use all 397 boards. "
    "Runtime/deployment repetition gates remain separate."
)
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
