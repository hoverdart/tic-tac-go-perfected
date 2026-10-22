"""Read-only archive audit and optional local solver baseline for future V4 work.

Run from the repository root: python -m scripts.audit_catalog --help.
No database credentials, writes, or Gemini calls are used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, datetime, UTC
import json
import platform
from pathlib import Path
import time
from urllib.request import Request, urlopen

from apps.api.puzzle_catalog import CatalogSnapshot, fetch_catalog, parse_catalog
from solver.push_solver import solve, verify_solution


def canonical_board(board):
    """Remove only fully blocked outer borders for layout comparisons."""
    rows = [i for i, row in enumerate(board) if any(cell != "B" for cell in row)]
    cols = [i for i in range(len(board[0])) if any(row[i] != "B" for row in board)]
    if not rows or not cols:
        raise ValueError("Board has no playable cells")
    return [list(row[min(cols):max(cols)+1]) for row in board[min(rows):max(rows)+1]]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_json(url):
    with urlopen(Request(url, headers={"User-Agent": "TicTacGo-ReadOnly-Audit"}), timeout=30) as response:
        return json.load(response)


def collect_records(api_base, through):
    history = read_json(f"{api_base}/solutions/recent?limit=10000")
    dates = [row["puzzle_date"] for row in history if date.fromisoformat(row["puzzle_date"]) <= through]
    # These are public GETs. Retries are safe and never trigger a daily job.
    def get_record(day):
        for attempt in range(3):
            try:
                row = read_json(f"{api_base}/solutions/{day}")
                if row.get("puzzle_date") != day:
                    raise ValueError(f"Archive response date mismatch for {day}")
                return {key: value for key, value in row.items() if key != "step_boards"}
            except Exception:
                if attempt == 2:
                    raise
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(get_record, dates))


def audit(snapshot: CatalogSnapshot, records, through):
    results, positives = [], []
    entries = [entry for day, entry in sorted(snapshot.entries.items()) if day <= through]
    manifest = [{"id": entry.puzzle_date.strftime("%Y%m%d"), "name": entry.title,
                 "date": {"year": entry.puzzle_date.year, "month": entry.puzzle_date.month, "day": entry.puzzle_date.day},
                 "width": entry.width, "height": entry.height, "puzzle": entry.level,
                 "board_sha256": entry.board_sha256} for entry in entries]
    for record in sorted(records, key=lambda row: row["puzzle_date"]):
        day = date.fromisoformat(record["puzzle_date"])
        if day > through:
            continue
        entry = snapshot.entries.get(day)
        result = {"date": day.isoformat(), "stored_status": record["status"],
                  "stored_title": record.get("puzzle_title"), "parser": record.get("parser_name"),
                  "solver": record.get("solver_name"), "elapsed_ms": record.get("elapsed_ms"),
                  "reported_states": record.get("states_checked"), "official_title": entry.title if entry else None}
        if entry is None:
            result.update(error="missing_catalog_date", mismatch=None)
            results.append(result)
            continue
        result["board_sha256"] = entry.board_sha256
        official = entry.board()
        board = record.get("board")
        if board is None:
            result.update(error="missing_stored_board", mismatch=True)
        else:
            old, new = canonical_board(board), canonical_board(official)
            result["mismatch"] = old != new
            if old != new:
                result.update(stored_board=board, official_board=official)
                if len(old) == len(new) and len(old[0]) == len(new[0]):
                    result["cell_differences"] = [
                        {"row": r+1, "col": c+1, "stored": old[r][c], "official": new[r][c]}
                        for r in range(len(old)) for c in range(len(old[0])) if old[r][c] != new[r][c]
                    ]
        if record["status"] == "solved":
            own = verify_solution(board, record.get("moves")) if board else None
            verified = verify_solution(official, record.get("moves"))
            result.update(valid_on_stored=bool(own and own.ok), valid_on_official=verified.ok,
                          verification_error=verified.error, stored_moves=record.get("moves"))
            if verified.ok:
                positives.append({**next(row for row in manifest if row["id"] == day.strftime("%Y%m%d")),
                                  "solution": record["moves"], "provenance": "stored_path_verified_on_catalog"})
        results.append(result)
    summary = {"stored": len(results), "marked_solved": sum(row["stored_status"] == "solved" for row in results),
               "mismatched_boards": sum(row["mismatch"] is True for row in results),
               "verified_official_paths": sum(row.get("valid_on_official", False) for row in results),
               "false_solved": sum(row.get("valid_on_official") is False for row in results),
               "unsolved": sum(row["stored_status"] != "solved" for row in results),
               "missing_stored_dates": [entry.puzzle_date.isoformat() for entry in entries
                                        if entry.puzzle_date.isoformat() not in {row['date'] for row in results}]}
    return manifest, results, positives, summary


def export_recovered(output_dir, snapshot, manifest):
    """Keep newly found paths separate from the original production receipts."""
    path = output_dir / "solver-runs.jsonl"
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    entries = {row["id"]: row for row in manifest}
    recovered = []
    for run in runs:
        day = date.fromisoformat(run["date"])
        entry = snapshot.entries[day]
        moves = run["result"]["moves"]
        if run["verified"] and verify_solution(entry.board(), moves).ok:
            recovered.append({**entries[day.strftime("%Y%m%d")], "solution": moves,
                              "provenance": "local_push_v3_recovery", "wall_seconds": run["wall_seconds"]})
    write_jsonl(output_dir / "recovered-solutions.jsonl", recovered)
    write_json(output_dir / "baseline-summary.json", {
        "targets": len(runs), "verified_recoveries": len(recovered),
        "remaining": [row["date"] for row in runs if not row["verified"]],
        "note": "Local sequential coverage runs; quality optimization disabled; no production records updated.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--through", type=date.fromisoformat, required=True, help="Exclude future boards from every training/benchmark export")
    parser.add_argument("--api-base-url", default="https://tictacgo.shauryav.com/api/python")
    parser.add_argument("--records", type=Path, help="Use a saved stored-records.jsonl instead of live public GETs")
    parser.add_argument("--catalog-source", type=Path, help="Parse a saved JS bundle instead of fetching")
    parser.add_argument("--solve-targets", action="store_true", help="Locally rerun all unsolved/invalid paths, sequentially")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--max-nodes", type=int, default=500000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = parse_catalog(args.catalog_source.read_text(), source_url=str(args.catalog_source)) if args.catalog_source else fetch_catalog()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()] if args.records else collect_records(args.api_base_url.rstrip('/'), args.through)
    records = [row for row in records if date.fromisoformat(row["puzzle_date"]) <= args.through]
    manifest, results, positives, summary = audit(snapshot, records, args.through)
    write_jsonl(args.output_dir / "catalog.jsonl", manifest)
    write_jsonl(args.output_dir / "stored-records.jsonl", records)
    write_jsonl(args.output_dir / "verified-solutions.jsonl", positives)
    write_json(args.output_dir / "archive-audit.json", results)
    write_json(args.output_dir / "mismatches.json", [row for row in results if row["mismatch"] is True])
    targets = [row for row in results if row.get("board_sha256") and (
        row["stored_status"] != "solved" or row.get("valid_on_official") is False)]
    write_json(args.output_dir / "targets.json", targets)
    metadata = {"captured_at": datetime.now(UTC).isoformat(), "through": args.through.isoformat(),
                "source_url": snapshot.source_url, "source_sha256": snapshot.source_sha256,
                "api_base_url": args.api_base_url, "python": platform.python_version(),
                "platform": platform.platform(), "summary": summary,
                "solver_budget": {"timeout_seconds": args.timeout_seconds, "max_nodes": args.max_nodes,
                                  "quality_timeout_seconds": 0, "execution": "sequential"},
                "limitations": ["Current published catalog, not archived screenshots", "Replay verification, not execution in Google's game",
                                "CNN reported states are replay transitions, not search expansions", "Future boards excluded"]}
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(summary), flush=True)
    if args.solve_targets:
        with (args.output_dir / "solver-runs.jsonl").open("w") as output:
            for row in targets:
                entry = snapshot.entries[date.fromisoformat(row["date"])]
                started = time.perf_counter()
                solved = solve(entry.board(), timeout_seconds=args.timeout_seconds, max_nodes=args.max_nodes, quality_timeout_seconds=0)
                run = {"date": row["date"], "title": entry.title, "board_sha256": entry.board_sha256,
                       "wall_seconds": time.perf_counter()-started, "result": asdict(solved),
                       "verified": verify_solution(entry.board(), solved.moves).ok if solved.moves is not None else False}
                output.write(json.dumps(run) + "\n")
                output.flush()
                print(f"{row['date']} solved={solved.solved} verified={run['verified']} seconds={run['wall_seconds']:.2f}", flush=True)
        export_recovered(args.output_dir, snapshot, manifest)


if __name__ == "__main__":
    main()
