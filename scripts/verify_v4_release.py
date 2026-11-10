"""Read-only verification of the deployed release kernel and its complete cohorts."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from apps.api.puzzle_catalog import CatalogEntry
from backfill_puzzle_titles import load_env_file
from solver.push_solver import verify_solution
from solver.push_solver.benchmark_v4 import (
    ENGINE_SHA256,
    RUNNER_SHA256,
    TARGETS,
    corpus,
    gate,
    provenance,
    read_rows,
    repetition_gate,
)

GUARDS = ("20260301", "20260802", "20261024")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "matrix", "guards", "corpus"), required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--routes", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    load_env_file(args.env_file)
    secret = os.environ["CRON_SECRET"]
    base = args.api_base.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def request(path, body=None):
        req = Request(
            base + path, data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
        )
        with urlopen(req, timeout=75) as response:
            return json.load(response)

    config = request("/debug/solver-config")
    if config["engine_sha256"] != ENGINE_SHA256 or config["runner_sha256"] != RUNNER_SHA256:
        raise SystemExit("Deployed kernel/runner differs from this release candidate")
    if config["provenance"]["sources"] != provenance()["sources"]:
        raise SystemExit("Deployed runtime sources differ from this release candidate")
    runtime = config["provenance"]["runtime"]
    native = runtime.get("native_build")
    if args.require_native and (
        runtime.get("implementation") != "cython" or not native
        or native.get("cython") != "3.3.1"
        or len(native.get("extensions", {})) != 2
        or native.get("sources") != {
            name: config["provenance"]["sources"]["solver/push_solver/" + name]
            for name in ("v4.py", "path_optimizer.py")
        }
    ):
        raise SystemExit("Expected the verified compiled production kernel")
    metadata = {
        "api_base": base, "mode": args.mode, "engine_sha256": ENGINE_SHA256,
        "runner_sha256": RUNNER_SHA256, "revision": config.get("revision"),
        "config": config, "started_at": datetime.now(UTC).isoformat(),
        "routes": args.routes,
    }
    meta_path = args.output_dir / (args.mode + ".metadata.json")
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if any(previous.get(key) != metadata[key] for key in ("api_base", "mode", "engine_sha256", "runner_sha256", "revision", "routes")):
            raise SystemExit("Use a new output directory for a different revision or configuration")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    catalog = corpus()
    entries = {row["id"]: row for row in catalog}
    if args.mode == "corpus":
        work = [(row["id"], "warm", 0) for row in catalog]
    else:
        ids = GUARDS if args.mode == "guards" else TARGETS
        if args.mode == "smoke":
            ids = (*TARGETS, *GUARDS)
        repeats = 10 if args.mode == "matrix" else 3 if args.mode == "guards" else 1
        processes = ("fresh", "warm") if args.mode in ("matrix", "guards") else ("warm",)
        work = [(key, process, repeat) for process in processes for repeat in range(repeats) for key in ids]
    output = args.output_dir / (args.mode + ".jsonl")
    rows = read_rows(output) if output.exists() else []
    existing = {(row["id"], row["process"], row["repeat"]) for row in rows}
    for key, process, repeat in work:
        if (key, process, repeat) in existing:
            continue
        entry = entries[key]
        started = time.perf_counter()
        try:
            receipt = request("/debug/push-benchmark/" + entry["date"], {
                "version": "v4", "schedule": "coverage", "process": process,
                "timeout_seconds": 30, "max_nodes": 500000, "quality_timeout_seconds": 30,
                "route_limit": args.routes,
            })
            if (
                receipt.get("engine_sha256") != ENGINE_SHA256
                or receipt.get("runner_sha256") != RUNNER_SHA256
                or receipt.get("board_sha256") != entry["board_sha256"]
                or receipt.get("id") != key or receipt.get("process") != process
                or receipt.get("revision") != config.get("revision")
                or receipt.get("quality_timeout_seconds") != 30
                or receipt.get("limits") != {"seconds": 30, "nodes": 500000}
                or receipt.get("schedule") != "coverage"
                or receipt.get("strategy_requested") is not None
                or receipt.get("routes") != args.routes
            ):
                raise ValueError("Mismatched deployment, board, process, or quality configuration")
            board = CatalogEntry(
                datetime.fromisoformat(entry["date"]).date(), entry["name"],
                entry["width"], entry["height"], entry["puzzle"],
            ).board()
            replay = verify_solution(board, receipt["result"]["moves"])
            solved_runtime = receipt["result"]["diagnostics"]["runtime"]
            if (
                solved_runtime.get("implementation") != runtime.get("implementation")
                or solved_runtime.get("native_build") != native
                or receipt["result"]["diagnostics"].get("route_limit") != args.routes
            ):
                raise ValueError("Worker implementation differs from the deployed build")
            if replay.ok != receipt["verified"]:
                raise ValueError("Independent replay disagrees with deployed verification")
            if replay.ok and replay.final_board != tuple(tuple(line) for line in receipt["result"]["final_board"]):
                raise ValueError("Deployed final board differs from independent replay")
            receipt["independently_verified"] = replay.ok
        except (HTTPError, OSError, ValueError, KeyError) as exc:
            receipt = {
                "id": key, "board_sha256": entry["board_sha256"], "version": "v4",
                "process": process, "verified": False, "wall_seconds": time.perf_counter() - started,
                "error": type(exc).__name__ + ": " + str(exc),
                "result": {"solved": False, "moves": None, "nodes_expanded": 0, "failure_reason": "verification_or_transport_failure"},
            }
        receipt["repeat"] = repeat
        receipt["http_wall_seconds"] = time.perf_counter() - started
        with output.open("a") as target:
            target.write(json.dumps(receipt) + "\n")
        rows.append(receipt)
        print(f"{len(rows)}/{len(work)} {key} {process} verified={receipt['verified']} seconds={receipt['wall_seconds']:.3f}", flush=True)
    if args.mode == "matrix":
        result = repetition_gate(rows)
    elif args.mode == "corpus":
        result = gate(rows, read_rows(Path(__file__).resolve().parents[1] / "debug-artifacts/v4-2026-10-25/v3-baseline.jsonl"), catalog)
        result["passed"] = result["passed"] and result["v4_verified"] == 397
    else:
        result = {
            "passed": len(rows) == len(work) and all(
                row["verified"] and row["wall_seconds"] <= 30
                and row["result"]["nodes_expanded"] <= 500000
                and (row["process"] != "fresh" or row.get("process_wall_seconds", 31) <= 30)
                for row in rows
            ),
            "receipts": len(rows), "expected": len(work),
        }
    (args.output_dir / (args.mode + ".gate.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
