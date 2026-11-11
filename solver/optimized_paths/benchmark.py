"""Reproducible offline optimizer cohorts; archive reuse is explicitly reported."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from hashlib import sha256
from pathlib import Path

from solver.push_solver.v4 import runtime_environment
from solver.push_solver.verify import verify_solution

from . import optimize
from .bank import ROOT, load_bank


def engine_sources():
    paths = [
        *Path(__file__).parent.glob("*.py"),
        ROOT / "solver/board_utils.py",
        ROOT / "solver/push_solver/models.py",
        ROOT / "solver/push_solver/verify.py",
        ROOT / "solver/push_solver/v4.py",
        ROOT / "solver/push_solver/path_optimizer.py",
        ROOT / "apps/api/puzzle_catalog.py",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("graph", "repair", "exact"), default="repair"
    )
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--max-layouts", type=int, default=8192)
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; preserve prior cohorts")
    intake_started = time.perf_counter()
    bank = load_bank(args.manifest)
    selected = sorted(bank.cases if args.ids is None else set(args.ids))
    if not selected or set(selected) - bank.cases.keys():
        parser.error("Unknown or empty board selection")
    options = {
        key: getattr(args, key)
        for key in ("mode", "seconds", "max_nodes", "radius", "max_layouts")
    }
    sources = engine_sources()
    metadata = {
        "schema": "optimized-path-benchmark-v1",
        "scope": "archive_seeded_optimization",
        "fresh_solver_benchmark": False,
        "options": options,
        "selected_ids": selected,
        "engine_sources": sources,
        "engine_sha256": sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        "inputs": bank.manifest,
        "intake": bank.counts,
        "rejected_rows": bank.rejected,
        "intake_ms": (time.perf_counter() - intake_started) * 1000,
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "reused_v4": runtime_environment(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    receipts = []
    with (args.output_dir / "results.jsonl").open("w") as handle:
        for key in selected:
            case = bank.cases[key]
            seeds = bank.paths[key]
            row = {
                "id": key,
                "name": case.name,
                "board_sha256": case.board_sha256,
                "seed_count": len(seeds),
                "engine_sha256": metadata["engine_sha256"],
            }
            if not seeds:
                row.update(status="missing_seed", verified=False)
            else:
                try:
                    result = optimize(case.board, seeds, **options)
                    if not verify_solution(case.board, result.moves).ok:
                        raise RuntimeError("Independent benchmark replay failed")
                    if len(result.moves) > min(map(len, seeds)):
                        raise RuntimeError("Optimizer regressed its shortest seed")
                    row.update(
                        status="verified", verified=True, result=result.to_dict()
                    )
                # Unexpected failures remain visible instead of dropping cases.
                except Exception as exc:  # noqa: BLE001
                    row.update(
                        status="error",
                        verified=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            receipts.append(row)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
    valid = [row["result"] for row in receipts if row["verified"]]
    summary = {
        "scope": metadata["scope"],
        "engine_sha256": metadata["engine_sha256"],
        "expected_boards": len(selected),
        "verified": len(valid),
        "errors": sum(row["status"] == "error" for row in receipts),
        "missing_seeds": sum(row["status"] == "missing_seed" for row in receipts),
        "complete": len(receipts) == len(selected),
        "all_verified": len(valid) == len(selected),
        "mean_seed_moves": statistics.mean(row["baseline_moves"] for row in valid)
        if valid
        else None,
        "mean_moves": statistics.mean(row["move_count"] for row in valid)
        if valid
        else None,
        "total_moves": sum(row["move_count"] for row in valid),
        "moves_saved": sum(row["baseline_moves"] - row["move_count"] for row in valid),
        "boards_improved": sum(row["improved"] for row in valid),
        "globally_proven": sum(row["optimality_proven"] for row in valid),
        "max_elapsed_ms": max((row["elapsed_ms"] for row in valid), default=None),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
