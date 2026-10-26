"""Aggregate PHS research receipts without combining configurations or budgets."""

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

from solver.push_solver.benchmark_v4 import (
    ENGINE_SHA256,
    TARGETS,
    corpus,
    gate,
    read_rows,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CATALOG = corpus()
BASELINE = read_rows(ROOT / "debug-artifacts/v4-2026-10-25/v3-baseline.jsonl")
REFERENCE = {row["id"]: row for row in BASELINE}


def daily_good(row):
    return bool(
        row.get("verified")
        and row["wall_seconds"] <= 30
        and row["result"]["nodes_expanded"] <= 500000
        and row.get("limits") == {"seconds": 30, "nodes": 500000}
    )


def summarize(rows):
    good = [r for r in rows if daily_good(r)]
    diagnostics = [(r["result"].get("diagnostics") or {}) for r in rows]
    result = {
        "records": len(rows),
        "distinct_boards": len({r["id"] for r in rows}),
        "verified": sum(bool(r.get("verified")) for r in rows),
        "within_daily_budget": len(good),
        "failed_daily_ids": sorted({r["id"] for r in rows if not daily_good(r)}),
        "matches_current_engine": all(
            r.get("engine_sha256") == ENGINE_SHA256 for r in rows
        ),
        "configuration": {
            "schedules": sorted({r.get("schedule", "") for r in rows}),
            "strategies": sorted({r.get("strategy_requested") or "" for r in rows}),
            "model_hashes": sorted(
                {
                    json.dumps(d.get("model_hashes", {}), sort_keys=True)
                    for d in diagnostics
                }
            ),
        },
        "effective_schedules": dict(
            Counter(d.get("effective_schedule") for d in diagnostics)
        ),
        "median_solved_wall_seconds": median(r["wall_seconds"] for r in good)
        if good
        else None,
        "process_peak_rss_bytes": max(
            (d.get("runtime", {}).get("peak_rss_bytes", 0) for d in diagnostics),
            default=0,
        ),
        "targets_with_any_daily_success": sum(
            any(r["id"] == key and daily_good(r) for r in rows) for key in TARGETS
        ),
        "v3_losses_present_in_cohort": sorted(
            {
                r["id"]
                for r in rows
                if REFERENCE[r["id"]]["verified"] and not daily_good(r)
            }
        ),
    }
    if len(rows) == 397 and len({r["id"] for r in rows}) == 397:
        result["current_engine_coverage_gate"] = gate(rows, BASELINE, CATALOG)
        common = [r for r in good if REFERENCE[r["id"]]["verified"]]
        result["common_v3_solved_moves"] = {
            "boards": len(common),
            "v3_total": sum(len(REFERENCE[r["id"]]["result"]["moves"]) for r in common),
            "candidate_total": sum(len(r["result"]["moves"]) for r in common),
            "v3_median": median(
                len(REFERENCE[r["id"]]["result"]["moves"]) for r in common
            ),
            "candidate_median": median(len(r["result"]["moves"]) for r in common),
        }
    return result


summary = {
    "implementation_commit": "04d5cf560223e5be25f340675d645e39cbbd7df6",
    "current_engine_sha256": ENGINE_SHA256,
    "scope": "Frozen 397-board development corpus, no fallback or exact action hints. Each corpus sequential; independent native experiments concurrent. Intermediate fingerprints stay separate. Native and local ARM64 container results are not deployment-matched acceptance.",
    "cohorts": {},
}
for path in sorted(HERE.glob("*.jsonl")):
    summary["cohorts"][path.name] = summarize(read_rows(path))
summary["recovered_repeats"] = summarize(
    read_rows(HERE / "recovered-fresh.jsonl") + read_rows(HERE / "recovered-warm.jsonl")
)
summary["deployed"] = {}
for path in sorted(HERE.glob("deployed-star-*.json")):
    receipt = json.loads(path.read_text())
    result = receipt.get("response", {}).get("result", {})
    summary["deployed"][path.name] = {
        "http_status": receipt["http_status"],
        "verified": receipt.get("response", {}).get("verified"),
        "matches_current_engine": receipt["matches_current_engine"],
        "matches_current_model": receipt["matches_current_model"],
        "expansions": result.get("nodes_expanded"),
        "elapsed_ms": result.get("elapsed_ms"),
        "cpu_ms": (result.get("diagnostics") or {}).get("cpu_ms"),
    }
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

control = {r["id"]: r for r in read_rows(HERE / "corpus-control-frozen.jsonl")}
adaptive = {r["id"]: r for r in read_rows(HERE / "corpus-adaptive.jsonl")}
with (HERE / "control-vs-adaptive.csv").open("w", newline="") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "id",
            "name",
            "board_sha256",
            "control_verified",
            "control_seconds",
            "control_nodes",
            "control_moves",
            "adaptive_verified",
            "adaptive_seconds",
            "adaptive_nodes",
            "adaptive_moves",
            "adaptive_effective_schedule",
            "adaptive_commitments",
        ]
    )
    for entry in CATALOG:
        a, b = control[entry["id"]], adaptive[entry["id"]]
        writer.writerow(
            [
                entry["id"],
                entry["name"],
                entry["board_sha256"],
                daily_good(a),
                a["wall_seconds"],
                a["result"]["nodes_expanded"],
                len(a["result"]["moves"] or ""),
                daily_good(b),
                b["wall_seconds"],
                b["result"]["nodes_expanded"],
                len(b["result"]["moves"] or ""),
                b["result"]["diagnostics"]["effective_schedule"],
                b["result"]["diagnostics"]["initial_commitment_count"],
            ]
        )
print(
    json.dumps(
        {
            name: {
                "daily": row["within_daily_budget"],
                "targets": row["targets_with_any_daily_success"],
                "current": row["matches_current_engine"],
            }
            for name, row in summary["cohorts"].items()
            if name.startswith("corpus-")
        },
        indent=2,
    )
)
