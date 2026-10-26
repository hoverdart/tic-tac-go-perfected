"""Regenerate round-two metrics without combining different search policies."""

import json
from hashlib import sha256
from pathlib import Path
from statistics import median

from solver.push_solver.benchmark_v4 import (
    TARGETS,
    corpus,
    gate,
    read_rows,
    repetition_gate,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
engine = sha256((ROOT / "solver/push_solver/v4.py").read_bytes()).hexdigest()
baseline = read_rows(ROOT / "debug-artifacts/v4-2026-10-25/v3-baseline.jsonl")
reference = {r["id"]: r for r in baseline}


def within_budget(row):
    return bool(
        row["verified"]
        and row["wall_seconds"] <= 30
        and row["result"]["nodes_expanded"] <= 500000
    )


def summarize(rows):
    good = [r for r in rows if within_budget(r)]
    common = [r for r in good if reference[r["id"]]["verified"]]
    return {
        "records": len(rows),
        "distinct_boards": len({r["id"] for r in rows}),
        "verified": sum(r["verified"] for r in rows),
        "verified_within_budget": len(good),
        "failed_ids": sorted({r["id"] for r in rows if not within_budget(r)}),
        "coverage_losses_present_in_cohort": sorted(
            {
                r["id"]
                for r in rows
                if reference[r["id"]]["verified"] and not within_budget(r)
            }
        ),
        "matches_current_engine": all(r.get("engine_sha256") == engine for r in rows)
        if rows[0]["version"] == "v4"
        else None,
        "solved_median_seconds": median(r["wall_seconds"] for r in good)
        if good
        else None,
        "max_wall_seconds": max(r["wall_seconds"] for r in rows),
        "process_peak_rss_bytes": max(
            (
                (r["result"].get("diagnostics") or {})
                .get("runtime", {})
                .get("peak_rss_bytes", 0)
                for r in rows
            ),
            default=0,
        ),
        "common_v3_solved_move_comparison": {
            "records": len(common),
            "median_v3_moves": median(
                len(reference[r["id"]]["result"]["moves"]) for r in common
            )
            if common
            else None,
            "median_candidate_moves": median(len(r["result"]["moves"]) for r in common)
            if common
            else None,
            "total_v3_moves": sum(
                len(reference[r["id"]]["result"]["moves"]) for r in common
            ),
            "total_candidate_moves": sum(len(r["result"]["moves"]) for r in common),
        },
        "targets": {
            key: {
                "runs": len(selected := [r for r in rows if r["id"] == key]),
                "verified_within_budget": sum(within_budget(r) for r in selected),
                "min_seconds": min((r["wall_seconds"] for r in selected), default=None),
                "max_seconds": max((r["wall_seconds"] for r in selected), default=None),
            }
            for key in TARGETS
        },
    }


output = {
    "engine_sha256": engine,
    "implementation_commit": "be102877d0732aeada084d7ad50576596d83a4c3",
    "scope": "Native Mac corpora and local Linux ARM64 container probes are research evidence. Each corpus ran sequentially, with independent experiments concurrently on the host. No deployment-matched 10 fresh + 10 warm repetition gate or unseen-board claim.",
    "production_default": "v3",
    "cohorts": {},
}
for path in sorted(HERE.glob("*.jsonl")):
    rows = read_rows(path)
    if not rows:
        continue
    result = summarize(rows)
    if path.name.startswith("corpus-") and len(rows) == 397:
        result["coverage_gate"] = gate(rows, baseline, corpus())
    if path.name.startswith("targets-") and (
        "fresh" in path.name or "warm" in path.name
    ):
        result["required_180_run_gate"] = repetition_gate(rows)
    output["cohorts"][path.name] = result
output["native_repetitions"] = {}
for stem in ("targets-beam8", "targets-beam8-straight"):
    rows = read_rows(HERE / (stem + "-fresh.jsonl")) + read_rows(
        HERE / (stem + "-warm.jsonl")
    )
    output["native_repetitions"][stem] = {
        **summarize(rows),
        "required_180_run_gate": repetition_gate(rows),
    }
(HERE / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps(
        {
            k: (v["records"], v["verified_within_budget"])
            for k, v in output["cohorts"].items()
            if k.startswith("corpus-")
        },
        indent=2,
    )
)
