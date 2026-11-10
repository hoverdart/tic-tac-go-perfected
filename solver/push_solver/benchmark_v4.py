"""Frozen-corpus comparisons and reproducible V4 release receipts (no fallback).

python -m solver.push_solver.benchmark_v4 --output /tmp/v4.jsonl --targets
python -m solver.push_solver.benchmark_v4 --output /tmp/v4.jsonl --gate-v3 ...
All rows include original board identity, budgets, source/model hashes, and replay
verification. Saved paths are evidence only; V4 never loads action hints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import date
from hashlib import sha256
from pathlib import Path

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver import solve_v3, solve_v4, verify_solution
from solver.push_solver.v4 import (
    EXPERIMENTAL_STRATEGIES,
    SCHEDULES,
    STRATEGIES,
    runtime_environment,
)

ROOT = Path(__file__).resolve().parents[2]
ENGINE_SOURCES = {
    name: sha256((ROOT / "solver/push_solver" / name).read_bytes()).hexdigest()
    for name in ("v4.py", "path_optimizer.py", "phs.py", "push_policy.py", "v4_ranker.py")
}
ENGINE_SHA256 = sha256(json.dumps(ENGINE_SOURCES, sort_keys=True).encode()).hexdigest()
RUNNER_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
CORPUS = ROOT / "debug-artifacts/v4-2026-10-25/catalog.jsonl"
TARGETS = (
    "20251212",
    "20251228",
    "20260314",
    "20260911",
    "20260913",
    "20261004",
    "20261017",
    "20261018",
    "20261025",
)


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def corpus(path=CORPUS):
    rows = read_rows(path)
    if len(rows) != 397 or len({r["id"] for r in rows}) != 397:
        raise ValueError("Release corpus must contain exactly 397 distinct boards")
    for row in rows:
        entry = CatalogEntry(
            date.fromisoformat(row["date"]),
            row["name"],
            row["width"],
            row["height"],
            row["puzzle"],
        )
        if entry.board_sha256 != row["board_sha256"] or row["id"] > "20261025":
            raise ValueError(f"Invalid frozen board: {row['id']}")
    return rows


def provenance():
    sources = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "solver/push_solver").glob("*.py"))
    }
    models = {
        p.name: sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "solver/push_solver").glob("linear*.json"))
    }
    return {"sources": sources, "models": models, "runtime": runtime_environment()}


def row_for_entry(entry):
    return {
        "id": entry.puzzle_date.strftime("%Y%m%d"),
        "date": entry.puzzle_date.isoformat(), "name": entry.title,
        "width": entry.width, "height": entry.height, "puzzle": entry.level,
        "board_sha256": entry.board_sha256,
    }


def run_fresh(row, **options):
    """Run only our allowlisted worker; input is JSON, never shell code."""
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "solver.push_solver.benchmark_v4", "--worker", "--worker-input", "--output", "-"],
        input=json.dumps({"row": row, "options": options}),
        capture_output=True, text=True, check=True, timeout=options.get("seconds", 30),
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), *(str(path) for path in sys.path if path)])},
    )
    receipt = json.loads(completed.stdout)
    receipt["process_wall_seconds"] = time.perf_counter() - started
    expected = {
        "limits": {"seconds": options.get("seconds", 30), "nodes": options.get("nodes", 500000)},
        "quality_timeout_seconds": options.get("quality_seconds", 0),
        "schedule": options.get("schedule", "rank"),
        "strategy_requested": options.get("strategy"),
    }
    if (
        receipt.get("id") != row["id"]
        or receipt.get("board_sha256") != row["board_sha256"]
        or receipt.get("version") != options.get("version", "v4")
        or receipt.get("runner_sha256") != RUNNER_SHA256
        or receipt.get("process") != "fresh"
        or any(receipt.get(key) != value for key, value in expected.items())
        or not isinstance(receipt.get("result"), dict)
    ):
        raise ValueError("Fresh worker returned mismatched provenance")
    if receipt["version"] == "v4" and receipt.get("engine_sha256") != ENGINE_SHA256:
        raise ValueError("Fresh worker returned a different engine")
    return receipt


def run(
    row,
    *,
    version="v4",
    seconds=30.0,
    nodes=500000,
    schedule="rank",
    routes=3,
    strategy=None,
    ranker_path=None,
    beam_mode="interleaved",
    macro_mode="straight",
    no_exact_hints=False,
    policy_path=None,
    repeat=0,
    process="warm",
    quality_seconds=0.0,
):
    board = CatalogEntry(
        date.fromisoformat(row["date"]),
        row["name"],
        row["width"],
        row["height"],
        row["puzzle"],
    ).board()
    kwargs = {
        "timeout_seconds": seconds,
        "max_nodes": nodes,
        "quality_timeout_seconds": quality_seconds,
    }
    if version == "v4":
        kwargs.update(
            schedule=schedule,
            route_limit=routes,
            strategy=strategy,
            ranker_path=ranker_path,
            beam_mode=beam_mode,
            macro_mode=macro_mode,
            policy_path=policy_path,
        )
    started = time.perf_counter()
    cpu = time.process_time()
    guard = nullcontext()
    if version == "v3" and no_exact_hints:
        from dataclasses import replace
        from unittest.mock import patch

        from solver.push_solver.rank_policy import (
            EnsemblePushRankPolicy,
            default_policy,
        )

        policy = default_policy()
        if isinstance(policy, EnsemblePushRankPolicy):
            policy = EnsemblePushRankPolicy(
                tuple(
                    replace(p, state_action_hints={}, priority_recovery=False)
                    for p in policy.policies
                )
            )
        elif policy is not None:
            policy = replace(policy, state_action_hints={}, priority_recovery=False)
        guard = patch(
            "solver.push_solver.rank_policy.default_policy", return_value=policy
        )
    with guard:
        result = (solve_v4 if version == "v4" else solve_v3)(board, **kwargs)
    verified = bool(result.solved and verify_solution(board, result.moves).ok)
    return {
        "id": row["id"],
        "board_sha256": row["board_sha256"],
        "version": version,
        "engine_sha256": ENGINE_SHA256 if version == "v4" else None,
        "runner_sha256": RUNNER_SHA256,
        "schedule": schedule,
        "beam_mode": beam_mode if version == "v4" else None,
        "macro_mode": macro_mode if version == "v4" else None,
        "exact_hints_enabled": version == "v3" and not no_exact_hints,
        "routes": routes,
        "quality_timeout_seconds": quality_seconds,
        "strategy_requested": strategy,
        "repeat": repeat,
        "process": process,
        "limits": {"seconds": seconds, "nodes": nodes},
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
        "verified": verified,
        "result": asdict(result),
    }


def gate(rows, baseline, catalog):
    """Fail closed on incomplete, stale, or out-of-budget evidence."""
    expected = {r["id"]: r["board_sha256"] for r in catalog}
    valid = {r["id"]: r for r in rows if r["board_sha256"] == expected.get(r["id"])}
    reference = {
        r["id"]: r for r in baseline if r["board_sha256"] == expected.get(r["id"])
    }

    def good(r):
        return bool(
            r.get("verified")
            and r.get("version") == "v4"
            and r["result"]["solved"]
            and r["wall_seconds"] <= 30
            and r["result"]["nodes_expanded"] <= 500000
            and r["result"].get("diagnostics", {}).get("exact_hints") is False
        )

    losses = [
        k
        for k, r in reference.items()
        if r.get("verified") and not good(valid.get(k, {}))
    ]
    missing_targets = [k for k in TARGETS if not good(valid.get(k, {}))]
    missing = sorted(set(expected) - valid.keys())
    stale = sorted(r["id"] for r in rows if r.get("engine_sha256") != ENGINE_SHA256)
    configurations = {
        (
            r.get("schedule"),
            r.get("routes"),
            r.get("strategy_requested"),
            r.get("beam_mode"),
            r.get("macro_mode"),
            r.get("quality_timeout_seconds", 0.0),
            json.dumps(
                r.get("result", {}).get("diagnostics", {}).get("model_hashes", {}),
                sort_keys=True,
            ),
        )
        for r in rows
    }
    invalid_budgets = sorted(
        r["id"]
        for r in rows
        if r.get("limits") != {"seconds": 30, "nodes": 500000}
        or r.get("strategy_requested") is not None
    )
    duplicate_or_extra_records = (
        len(rows) != 397 or len(valid) != 397 or bool(set(valid) - set(expected))
    )
    return {
        "passed": not (
            losses
            or missing_targets
            or missing
            or stale
            or invalid_budgets
            or len(configurations) != 1
            or len(reference) != 397
            or duplicate_or_extra_records
        ),
        "v3_verified": sum(bool(r.get("verified")) for r in reference.values()),
        "v4_verified": sum(bool(good(r)) for r in valid.values()),
        "coverage_losses": losses,
        "unsolved_targets": missing_targets,
        "missing": missing,
        "baseline_boards": len(reference),
        "stale_engine_inputs": stale,
        "invalid_budgets_or_individual_strategies": invalid_budgets,
        "configuration_count": len(configurations),
        "invalid_record_count": duplicate_or_extra_records,
        "scope": "Single corpus coverage gate only; container repetitions and hosted/deployed checks remain separate.",
    }


def repetition_gate(rows):
    hashes = {r["id"]: r["board_sha256"] for r in corpus()}
    expected = {
        (board, process, repeat)
        for board in TARGETS
        for process in ("fresh", "warm")
        for repeat in range(10)
    }
    found = set()
    failures = []
    configurations = set()
    for r in rows:
        key = r["id"], r.get("process"), r.get("repeat")
        if key not in expected:
            failures.append(key)
            continue
        if key in found:
            failures.append(key)
        found.add(key)
        configurations.add(
            (
                r.get("schedule"),
                r.get("routes"),
                r.get("strategy_requested"),
                r.get("beam_mode"),
                r.get("macro_mode"),
                r.get("quality_timeout_seconds", 0.0),
                json.dumps(
                    r.get("result", {}).get("diagnostics", {}).get("model_hashes", {}),
                    sort_keys=True,
                ),
            )
        )
        if not (
            r.get("verified")
            and r.get("engine_sha256") == ENGINE_SHA256
            and r.get("runner_sha256") == RUNNER_SHA256
            and r.get("board_sha256") == hashes[r["id"]]
            and r.get("limits") == {"seconds": 30, "nodes": 500000}
            and r.get("strategy_requested") is None
            and r["wall_seconds"] <= 30
            and (
                r.get("process") != "fresh"
                or 0 < r.get("process_wall_seconds", 0) <= 30
            )
            and r["result"]["nodes_expanded"] <= 500000
            and r.get("version") == "v4"
            and r["result"].get("diagnostics", {}).get("exact_hints") is False
        ):
            failures.append(key)
    return {
        "passed": found == expected and not failures and len(configurations) == 1,
        "missing": sorted(expected - found),
        "failures": failures,
        "configuration_count": len(configurations),
        "scope": "180 search receipts; deployment/container equivalence and hosted checks must also be verified.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--version", choices=["v3", "v4"], default="v4")
    p.add_argument("--targets", action="store_true")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--nodes", type=int, default=500000)
    p.add_argument("--quality-seconds", type=float, default=0.0)
    p.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="rank",
    )
    p.add_argument("--routes", type=int, default=3)
    p.add_argument(
        "--ranker",
        type=Path,
        help="Offline feature-only candidate; never enabled by the daily selector",
    )
    p.add_argument("--strategy", choices=EXPERIMENTAL_STRATEGIES)
    p.add_argument(
        "--policy",
        type=Path,
        help="Explicit feature-only probability model for PHS/Levin experiments",
    )
    p.add_argument(
        "--beam-mode", choices=["serial", "interleaved"], default="interleaved"
    )
    p.add_argument("--macro-mode", choices=["straight", "forced"], default="straight")
    p.add_argument(
        "--no-exact-hints",
        action="store_true",
        help="V3 comparison only; keep feature weights, remove every exact action hint",
    )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--fresh-process", action="store_true")
    p.add_argument(
        "--sweep",
        action="store_true",
        help="300s/5M per strategy; stop a board on a verified path",
    )
    p.add_argument("--gate-v3", type=Path)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-input", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker_input:
        if not a.worker:
            p.error("worker-input requires worker mode")
        payload = json.load(sys.stdin)
        row = payload["row"]
        entry = CatalogEntry(date.fromisoformat(row["date"]), row["name"], row["width"], row["height"], row["puzzle"])
        if row_for_entry(entry) != row:
            p.error("worker input does not match its official board identity")
        print(json.dumps(run(row, process="fresh", **payload["options"])))
        return
    if not 0 <= a.quality_seconds <= a.seconds:
        p.error("quality-seconds must be between zero and the total time budget")
    if a.policy and (a.version != "v4" or a.sweep):
        p.error(
            "Select a V4 policy strategy/schedule explicitly; policy sweeps use separate benchmark files"
        )
    catalog = corpus(a.corpus)
    selected = [
        r
        for r in catalog
        if (not a.targets or r["id"] in TARGETS) and (not a.ids or r["id"] in a.ids)
    ]
    if not selected:
        p.error("No boards selected")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.worker:
        print(
            json.dumps(
                run(
                    selected[0],
                    version=a.version,
                    seconds=a.seconds,
                    nodes=a.nodes,
                    schedule=a.schedule,
                    routes=a.routes,
                    strategy=a.strategy,
                    ranker_path=a.ranker,
                    beam_mode=a.beam_mode,
                    macro_mode=a.macro_mode,
                    no_exact_hints=a.no_exact_hints,
                    policy_path=a.policy,
                    process="fresh",
                    quality_seconds=a.quality_seconds,
                )
            )
        )
        return
    meta = a.output.with_suffix(".metadata.json")
    now = provenance()
    now["engine_sources"] = ENGINE_SOURCES
    now["experiment"] = {
        "version": a.version,
        "schedule": a.schedule,
        "strategy": a.strategy,
        "beam_mode": a.beam_mode,
        "macro_mode": a.macro_mode,
        "seconds": a.seconds,
        "nodes": a.nodes,
        "routes": a.routes,
        "quality_seconds": a.quality_seconds,
        "no_exact_hints": a.no_exact_hints,
        "policy_sha256": sha256(a.policy.read_bytes()).hexdigest()
        if a.policy
        else None,
        "ranker_sha256": sha256(a.ranker.read_bytes()).hexdigest()
        if a.ranker
        else None,
        "fresh_process": a.fresh_process,
        "sweep": a.sweep,
    }
    if meta.exists() and (
        json.loads(meta.read_text())["sources"] != now["sources"]
        or json.loads(meta.read_text()).get("experiment") != now["experiment"]
    ):
        p.error(
            "Source or configuration changed: use a new output path rather than mixing experiments"
        )
    meta.write_text(json.dumps(now, indent=2) + "\n")
    existing = read_rows(a.output) if a.output.exists() else []
    keys = {
        (r["id"], r.get("repeat", 0), r.get("strategy_requested")) for r in existing
    }
    for repeat in range(a.repeats):
        for row in selected:
            if a.sweep and any(
                r["id"] == row["id"] and r["verified"] for r in existing
            ):
                continue
            for strategy in STRATEGIES if a.sweep else (a.strategy,):
                if (row["id"], repeat, strategy) in keys:
                    continue
                seconds, nodes = (300, 5000000) if a.sweep else (a.seconds, a.nodes)
                if a.fresh_process:
                    cmd = [
                        sys.executable,
                        "-m",
                        "solver.push_solver.benchmark_v4",
                        "--worker",
                        "--output",
                        str(a.output),
                        "--corpus",
                        str(a.corpus),
                        "--ids",
                        row["id"],
                        "--version",
                        a.version,
                        "--seconds",
                        str(seconds),
                        "--nodes",
                        str(nodes),
                        "--schedule",
                        a.schedule,
                        "--routes",
                        str(a.routes),
                    ]
                    cmd += ["--beam-mode", a.beam_mode, "--macro-mode", a.macro_mode]
                    cmd += ["--quality-seconds", str(a.quality_seconds)]
                    if a.no_exact_hints:
                        cmd += ["--no-exact-hints"]
                    if strategy:
                        cmd += ["--strategy", strategy]
                    if a.ranker:
                        cmd += ["--ranker", str(a.ranker)]
                    if a.policy:
                        cmd += ["--policy", str(a.policy)]
                    worker_started = time.perf_counter()
                    try:
                        result = json.loads(
                            subprocess.check_output(
                                cmd, text=True, timeout=seconds + 15
                            )
                        )
                    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
                        result = {
                            "id": row["id"],
                            "board_sha256": row["board_sha256"],
                            "version": a.version,
                            "process": "fresh",
                            "verified": False,
                            "wall_seconds": time.perf_counter() - worker_started,
                            "strategy_requested": strategy,
                            "result": {
                                "solved": False,
                                "nodes_expanded": 0,
                                "failure_reason": "worker_failure",
                            },
                            "error": str(exc)[:1000],
                        }
                    result["process_wall_seconds"] = (
                        time.perf_counter() - worker_started
                    )
                    result["repeat"] = repeat
                else:
                    result = run(
                        row,
                        version=a.version,
                        seconds=seconds,
                        nodes=nodes,
                        schedule=a.schedule,
                        routes=a.routes,
                        strategy=strategy,
                        ranker_path=a.ranker,
                        beam_mode=a.beam_mode,
                        macro_mode=a.macro_mode,
                        no_exact_hints=a.no_exact_hints,
                        policy_path=a.policy,
                        repeat=repeat,
                        quality_seconds=a.quality_seconds,
                    )
                with a.output.open("a") as out:
                    out.write(json.dumps(result) + "\n")
                existing.append(result)
                print(
                    f"{row['id']} {strategy or a.schedule} solved={result['verified']} seconds={result['wall_seconds']:.3f} nodes={result['result']['nodes_expanded']}",
                    flush=True,
                )
                if a.sweep and result["verified"]:
                    break
    if a.gate_v3:
        report = gate(existing, read_rows(a.gate_v3), catalog)
        a.output.with_suffix(".gate.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
