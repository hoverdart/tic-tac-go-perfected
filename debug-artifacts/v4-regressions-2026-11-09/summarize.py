"""Rebuild the final native coverage summary and independently replay its paths."""

import json
from datetime import date
from pathlib import Path
from statistics import median

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver import verify_solution
from solver.push_solver.benchmark_v4 import corpus, gate, read_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
catalog = corpus()
entries = {row["id"]: row for row in catalog}
rows = read_rows(HERE / "corpus-coverage.jsonl")
if len(rows) != 397:
    raise SystemExit("Wait for all 397 corpus records before rebuilding the summary.")
baseline = read_rows(ROOT / "debug-artifacts/v4-2026-10-25/v3-baseline.jsonl")
reference = {row["id"]: row for row in baseline}
decision = gate(rows, baseline, catalog)
replayed = 0
for row in rows:
    entry = entries[row["id"]]
    board = CatalogEntry(
        date.fromisoformat(entry["date"]), entry["name"],
        entry["width"], entry["height"], entry["puzzle"],
    ).board()
    checked = verify_solution(board, row["result"]["moves"])
    assert checked.ok == row["verified"], row["id"]
    replayed += checked.ok

def within_budget(row):
    return row["verified"] and row["wall_seconds"] <= 30 and row["result"]["nodes_expanded"] <= 500000


common = [row for row in rows if within_budget(row) and reference[row["id"]]["verified"]]
v3_moves = [len(reference[row["id"]]["result"]["moves"]) for row in common]
v4_moves = [len(row["result"]["moves"]) for row in common]


def named(ids):
    return [{"id": key, "name": entries[key]["name"]} for key in ids]


def selected(file):
    path = HERE / file
    if not path.exists():
        return []
    return [
        {"id": row["id"], "name": entries[row["id"]]["name"],
         "verified": row["verified"], "wall_seconds": row["wall_seconds"],
         "moves": len(row["result"]["moves"] or ""),
         "nodes": row["result"]["nodes_expanded"],
         "cpu_ms": row["result"]["diagnostics"]["cpu_ms"]}
        for row in read_rows(path)
    ]


summary = {
    "gate": decision,
    "independently_replayed_solutions": replayed,
    "coverage_losses": named(decision["coverage_losses"]),
    "unsolved_targets": named(decision["unsolved_targets"]),
    "gains_over_v3": named([
        row["id"] for row in rows if within_budget(row) and not reference[row["id"]]["verified"]
    ]),
    "total_native_wall_seconds": sum(row["wall_seconds"] for row in rows),
    "max_native_wall_seconds": max(row["wall_seconds"] for row in rows),
    "peak_process_rss_bytes": max(
        row["result"]["diagnostics"]["runtime"]["peak_rss_bytes"] for row in rows
    ),
    "shared_v3_success_move_quality": {
        "boards": len(common),
        "v3_median": median(v3_moves), "v4_median": median(v4_moves),
        "v3_total": sum(v3_moves), "v4_total": sum(v4_moves),
        "total_change_percent": 100 * (sum(v4_moves) / sum(v3_moves) - 1),
        "quality_optimization": "disabled for both versions",
    },
    "fresh_native_cases": selected("focused-coverage.jsonl"),
    "linux_cases": selected("container-coverage.jsonl"),
    "scope": "Native development corpus only; deployment/repetition gates remain separate.",
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
