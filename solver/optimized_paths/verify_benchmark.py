"""Independently replay a complete cohort and reject mixed inputs/code or claims."""

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path

from solver.push_solver.verify import verify_solution

from .bank import ROOT, load_bank
from .benchmark import engine_sources


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_cohort(output_dir, manifest, *, root=ROOT):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
    ]
    bank = load_bank(manifest, root=root)
    require(
        metadata["engine_sources"] == engine_sources(), "Source fingerprints changed"
    )
    require(
        metadata["engine_sha256"]
        == sha256(
            json.dumps(metadata["engine_sources"], sort_keys=True).encode()
        ).hexdigest(),
        "Engine hash differs",
    )
    require(
        summary["engine_sha256"] == metadata["engine_sha256"]
        and summary["scope"] == metadata["scope"],
        "Mixed summary identity",
    )
    require(metadata["inputs"] == bank.manifest, "Bank inputs changed")
    require(
        metadata["intake"] == bank.counts
        and metadata["rejected_rows"] == bank.rejected,
        "Intake accounting differs",
    )
    require(
        metadata["scope"] == "archive_seeded_optimization"
        and metadata["fresh_solver_benchmark"] is False,
        "Incorrect benchmark scope",
    )
    selected = metadata["selected_ids"]
    require(
        len(rows) == len(selected) == len(set(selected))
        and {row["id"] for row in rows} == set(selected),
        "Missing or duplicate cases",
    )
    total = seed_total = improved = proven = verified = 0
    for row in rows:
        key = row["id"]
        require(row["seed_count"] == len(bank.paths[key]), "Seed count differs")
        require(
            row["status"] in ("verified", "missing_seed", "error"),
            "Unknown case status",
        )
        require(
            row["board_sha256"] == bank.cases[key].board_sha256
            and row["engine_sha256"] == metadata["engine_sha256"],
            "Mixed case identity",
        )
        if row["status"] != "verified":
            require(row["verified"] is False, "Failed row claims verification")
            continue
        result = row["result"]
        checked = verify_solution(bank.cases[key].board, result["moves"])
        require(
            checked.ok and result["verified"] is True and row["verified"] is True,
            "Independent replay failed",
        )
        require(
            tuple(map(tuple, result["final_board"])) == checked.final_board,
            "Final board differs",
        )
        length = len(result["moves"])
        baseline = min(map(len, bank.paths[key]))
        require(
            length == result["move_count"] <= baseline == result["baseline_moves"],
            "Move count or non-regression claim differs",
        )
        lower = result["lower_bound"]
        require(type(lower) is int and 0 <= lower <= length, "Invalid global bound")
        require(
            result["optimality_proven"] == (lower == length),
            "Proof flag differs from bound",
        )
        require(
            not result["optimality_proven"]
            or length == 0
            or metadata["options"]["mode"] == "exact",
            "Restricted search claims global proof",
        )
        require(
            result["expanded"] <= metadata["options"]["max_nodes"],
            "Expansion cap exceeded",
        )
        require(result["improved"] == (length < baseline), "Improvement flag differs")
        verified += 1
        total += length
        seed_total += baseline
        improved += length < baseline
        proven += result["optimality_proven"]
    for key, value in {
        "expected_boards": len(selected),
        "verified": verified,
        "total_moves": total,
        "moves_saved": seed_total - total,
        "boards_improved": improved,
        "globally_proven": proven,
        "complete": True,
        "all_verified": verified == len(selected),
        "errors": sum(row["status"] == "error" for row in rows),
        "missing_seeds": sum(row["status"] == "missing_seed" for row in rows),
    }.items():
        require(summary[key] == value, f"Summary differs: {key}")
    if verified:
        require(
            math.isclose(summary["mean_moves"], total / verified)
            and math.isclose(summary["mean_seed_moves"], seed_total / verified),
            "Mean differs",
        )
    return {
        "cases": len(selected),
        "independently_verified": verified,
        "all_verified": verified == len(selected),
        "source_and_inputs_match": True,
        "claims_checked": True,
        "globally_proven_by_search": proven,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = verify_cohort(args.output_dir, args.manifest)
    print(json.dumps(result, indent=2))
    return 0 if result["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
