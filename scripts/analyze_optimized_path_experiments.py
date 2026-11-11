"""Compare frozen cohorts and check cross-run bounds and optimal substructure.

The combined bank is explicitly archive aggregation, not a single-budget solve.
Only configuration-bound v2 runs supply bounds for new derived certificates.
"""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path

from solver.legacy_solver import apply_single_move, solved
from solver.optimized_paths.bank import load_bank
from solver.optimized_paths.verify_benchmark import verify_cohort


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired(first, second):
    if first.keys() != second.keys():
        raise ValueError("Paired cohorts differ in board identities")
    differences = [
        len(second[key]["result"]["moves"]) - len(first[key]["result"]["moves"])
        for key in first
    ]
    return {
        "boards": len(first),
        "shorter": sum(value < 0 for value in differences),
        "equal": differences.count(0),
        "longer": sum(value > 0 for value in differences),
        "total_moves_saved": -sum(differences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("debug-artifacts/optimized-paths-2026-11-10")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    root = args.root
    extended = root / "extended"
    names = [
        "graph",
        "repair-1s",
        "exact-1s",
        "exact-10s",
        "repair-radius2-10s",
        "exact-60s-guards",
    ]
    cohorts = {}
    summaries = {}
    strongest = {}
    bound_sources = {}
    direct = {}
    for name in names:
        directory = extended / name
        verify_cohort(directory, root / "inputs.json")
        rows = {row["id"]: row for row in read_rows(directory / "results.jsonl")}
        cohorts[name] = rows
        summary = json.loads((directory / "summary.json").read_text())
        resource = (directory / "resources.txt").read_text()
        match = re.search(r"(\d+)\s+maximum resident set size", resource)
        summaries[name] = {
            **summary,
            "peak_rss_mib": int(match[1]) / 1024**2 if match else None,
        }
        for key, row in rows.items():
            result = row["result"]
            lower = result["lower_bound"]
            source = {
                "path": str(directory / "results.jsonl"),
                "configuration_sha256": summary["configuration_sha256"],
            }
            if lower > strongest.get(key, -1):
                strongest[key] = lower
                bound_sources[key] = [source]
            elif lower == strongest[key]:
                bound_sources[key].append(source)
            if result["optimality_proven"]:
                length = len(result["moves"])
                if key in direct and direct[key] != length:
                    raise AssertionError(f"Conflicting completed proofs: {key}")
                direct[key] = length
    verify_cohort(extended / "refined-graph", extended / "inputs-refined.json")
    refined_rows = read_rows(extended / "refined-graph/results.jsonl")
    bank = load_bank(extended / "inputs-refined.json")
    certificates = []
    for row in refined_rows:
        key = row["id"]
        moves = row["result"]["moves"]
        if strongest[key] > len(moves):
            raise AssertionError(
                f"A new verified path contradicts a global bound: {key}"
            )
        if key in direct and direct[key] != len(moves):
            raise AssertionError(f"A new path contradicts a completed proof: {key}")
        if strongest[key] == len(moves):
            certificates.append(
                {
                    "id": key,
                    "board_sha256": row["board_sha256"],
                    "solution": moves,
                    "lower_bound": strongest[key],
                    "proof_kind": "completed_search"
                    if key in direct
                    else "cross_run_bound_match",
                    "bound_sources": bound_sources[key],
                    "path_source": str(extended / "refined-graph/results.jsonl"),
                }
            )
    # Exact suffix distance is unique for a full state, including its player.
    labels = {}
    repeated = 0
    for row in certificates:
        board = bank.cases[row["id"]].board
        moves = row["solution"]
        for step in range(len(moves) + 1):
            remaining = len(moves) - step
            if board in labels:
                if labels[board] != remaining:
                    raise AssertionError("Conflicting exact suffix distances")
                repeated += 1
            labels[board] = remaining
            if solved(board) and remaining:
                raise AssertionError("An alleged shortest path continues after winning")
            if step < len(moves):
                board = apply_single_move(board, moves[step])
    checked_steps = matched_steps = 0
    for key, paths in bank.paths.items():
        for moves in paths:
            board = bank.cases[key].board
            for step in range(len(moves) + 1):
                checked_steps += 1
                if board in labels:
                    matched_steps += 1
                    if labels[board] > len(moves) - step:
                        raise AssertionError(
                            f"Archived suffix contradicts a certificate: {key}"
                        )
                if step < len(moves):
                    board = apply_single_move(board, moves[step])
    confirmed = set()
    for row in json.loads((root / "stress/corpus-oracle.json").read_text()):
        if row["status"] == "independently_confirmed":
            if row["id"] not in direct or direct[row["id"]] != row["moves"]:
                raise AssertionError("Independent unit-step confirmation differs")
            confirmed.add(row["id"])
    for name in ["exact-10s-independent", "exact-60s-guards-independent"]:
        for row in read_rows(extended / name / "results.jsonl"):
            if row["status"] == "confirmed":
                if direct.get(row["id"]) != row["claimed_moves"]:
                    raise AssertionError("Independent set-based confirmation differs")
                confirmed.add(row["id"])
    summary = {
        "scope": "cross_run_archive_analysis_not_single_budget_benchmark",
        "cohorts": summaries,
        "exact_1s_to_10s": paired(cohorts["exact-1s"], cohorts["exact-10s"]),
        "repair_radius1_to_radius2": paired(
            cohorts["repair-1s"], cohorts["repair-radius2-10s"]
        ),
        "refined_bank": json.loads(
            (extended / "refined-graph/summary.json").read_text()
        ),
        "completed_search_certificates": len(direct),
        "certificates_after_bound_matching": len(certificates),
        "new_cross_run_bound_matches": sum(
            row["proof_kind"] == "cross_run_bound_match" for row in certificates
        ),
        "unique_independently_confirmed": len(confirmed),
        "certified_suffix_states": len(labels),
        "repeated_suffix_states": repeated,
        "candidate_suffix_steps_checked": checked_steps,
        "candidate_steps_matching_certified_states": matched_steps,
        "no_bound_or_suffix_contradictions": True,
        "analysis_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "certified-paths.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in certificates)
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "cohorts"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
