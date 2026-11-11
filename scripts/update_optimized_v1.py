"""Build/publish optimized-v1's static feed locally. Never imports database code.

The optional publisher writes only to the isolated optimized-v1-data branch.
Existing search receipts remain immutable; fresh receipts are checkpointed per board.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

from apps.api.puzzle_catalog import CatalogEntry, fetch_catalog
from solver.legacy_solver import apply_single_move, solved
from solver.optimized_paths.bank import load_bank
from solver.optimized_v1 import SOLVER_NAME, optimize
from solver.push_solver.verify import verify_solution

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "optimized-v1-feed-v1"
BRANCH = "optimized-v1-data"


def now():
    return datetime.now(UTC).isoformat()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def first_win(board, moves):
    """Reject illegal/loss paths and end at the first win, including zero moves."""
    check = verify_solution(board, moves)
    if not check.ok:
        raise ValueError(f"Invalid solution: {check.error}")
    state = tuple(map(tuple, board))
    if solved(state):
        return ""
    for i, move in enumerate(moves, 1):
        state = apply_single_move(state, move)
        if solved(state):
            return moves[:i]
    raise ValueError("Solution has no winning state")


def make_record(entry, moves, *, proof=False, lower_bound=0, provenance=None):
    original = moves
    moves = first_win(entry.board(), moves)
    if proof and (lower_bound != len(moves) or moves != original):
        raise ValueError("Shortest-path claim contradicts verified path")
    return {
        "puzzle_date": entry.puzzle_date.isoformat(),
        "puzzle_title": entry.title,
        "board_sha256": entry.board_sha256,
        "board": entry.board(),
        "moves": moves,
        "solver_name": SOLVER_NAME,
        "optimality_proven": bool(proof),
        "lower_bound": lower_bound,
        "updated_at": now(),
        "provenance": provenance or {},
    }


def validate_record(record):
    if record["solver_name"] != SOLVER_NAME:
        raise ValueError("Unexpected solver")
    if date.fromisoformat(record["puzzle_date"]).isoformat() != record["puzzle_date"]:
        raise ValueError("Invalid date")
    if first_win(record["board"], record["moves"]) != record["moves"]:
        raise ValueError("Path continues beyond a win")
    bound = record["lower_bound"]
    if type(bound) is not int or not 0 <= bound <= len(record["moves"]):
        raise ValueError("Invalid lower bound")
    if record["optimality_proven"] and bound != len(record["moves"]):
        raise ValueError("Invalid shortest-path claim")


def choose_record(previous, candidate):
    validate_record(candidate)
    if previous is None or previous["board_sha256"] != candidate["board_sha256"]:
        return candidate
    validate_record(previous)
    if previous["board"] != candidate["board"]:
        raise ValueError("Identical board hashes have different geometry")
    if previous["optimality_proven"] and len(candidate["moves"]) < len(
        previous["moves"]
    ):
        raise ValueError("New path contradicts an existing certificate")
    if candidate["optimality_proven"] and len(previous["moves"]) < len(
        candidate["moves"]
    ):
        raise ValueError("New certificate contradicts an existing path")
    if len(candidate["moves"]) < len(previous["moves"]):
        return candidate
    if len(candidate["moves"]) == len(previous["moves"]) and (
        candidate["optimality_proven"] and not previous["optimality_proven"]
    ):
        return candidate
    return previous


def read_feed(path):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if payload["schema"] != SCHEMA:
        raise ValueError("Invalid feed schema")
    records = {}
    for row in payload["solutions"]:
        validate_record(row)
        key = row["puzzle_date"]
        if key in records:
            raise ValueError("Duplicate feed date")
        records[key] = row
    return records


def save_feed(path, records):
    atomic_json(
        path,
        {
            "schema": SCHEMA,
            "solver_name": SOLVER_NAME,
            "solutions": [records[key] for key in sorted(records)],
        },
    )


def bootstrap(manifest, certificates):
    bank = load_bank(manifest)
    if bank.rejected:
        raise ValueError("Bootstrap archive contains rejected paths")
    claims = {}
    if certificates:
        for line in certificates.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in claims or row["proof_kind"] != "completed_search":
                raise ValueError("Invalid bootstrap certificate")
            claims[row["id"]] = row
    records = {}
    for key, case in bank.cases.items():
        if not bank.paths[key]:
            continue
        claim = claims.get(key)
        moves = min(bank.paths[key], key=lambda s: (len(s), s))
        # Equal-length alternative paths share the same minimum distance.
        if claim and not (
            claim["board_sha256"] == case.board_sha256
            and len(claim["solution"]) == len(moves)
            and claim["lower_bound"] == len(moves)
            and verify_solution(case.board, claim["solution"]).ok
        ):
            raise ValueError(f"Certificate mismatch: {key}")
        # Geometry is already validated by load_bank; retain the published hash.
        row = {
            "puzzle_date": date(int(key[:4]), int(key[4:6]), int(key[6:])).isoformat(),
            "puzzle_title": case.name,
            "board_sha256": case.board_sha256,
            "board": list(map(list, case.board)),
            "moves": first_win(case.board, moves),
            "solver_name": SOLVER_NAME,
            "optimality_proven": claim is not None,
            "lower_bound": claim["lower_bound"] if claim else 0,
            "updated_at": now(),
            "provenance": {"archive": bank.paths[key][moves], "certificate": claim},
        }
        validate_record(row)
        records[row["puzzle_date"]] = row
    if set(claims) - bank.cases.keys():
        raise ValueError("Unknown bootstrap certificates")
    return records


def load_entries(path=None):
    if path is None:
        return fetch_catalog().entries
    entries = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        d = date.fromisoformat(row["date"])
        entry = CatalogEntry(d, row["name"], row["width"], row["height"], row["puzzle"])
        if d in entries or row["board_sha256"] != entry.board_sha256:
            raise ValueError("Catalog identity mismatch")
        entries[d] = entry
    return entries


def solve_entry(entry, seconds, max_nodes):
    from solver.push_solver.v4 import solve_v4

    seed = solve_v4(
        entry.board(),
        schedule="coverage",
        route_limit=2,
        timeout_seconds=30,
        quality_timeout_seconds=30,
        max_nodes=500_000,
    )
    if seed.moves is None:
        raise ValueError("Local V4 did not find a verified seed")
    seeds = [first_win(entry.board(), seed.moves)]
    repair = optimize(
        entry.board(),
        seeds,
        mode="repair",
        radius=2,
        max_layouts=32768,
        seconds=seconds,
        max_nodes=max_nodes,
    )
    exact = optimize(
        entry.board(),
        [*seeds, repair.moves],
        mode="exact",
        seconds=seconds,
        max_nodes=max_nodes,
    )
    moves = min([*seeds, repair.moves, exact.moves], key=lambda s: (len(s), s))
    if exact.lower_bound > len(moves):
        raise ValueError("Search bound contradicts best verified path")
    receipt = {
        "solver_name": SOLVER_NAME,
        "board_sha256": entry.board_sha256,
        "seed": asdict(seed),
        "repair": repair.to_dict(),
        "exact": exact.to_dict(),
    }
    record = make_record(
        entry,
        moves,
        proof=exact.lower_bound == len(moves),
        lower_bound=exact.lower_bound,
        provenance={
            "kind": "local-v4-repair-exact",
            "seconds_per_phase": seconds,
            "max_nodes_per_phase": max_nodes,
        },
    )
    return record, receipt


def git(directory, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def publication_checkout(state):
    directory = state / "publication"
    if not (directory / ".git").exists():
        directory.mkdir(parents=True, exist_ok=True)
        git(directory, "init", "-b", BRANCH)
        git(
            directory,
            "remote",
            "add",
            "origin",
            git(ROOT, "remote", "get-url", "origin").stdout.strip(),
        )
        for key in ("user.name", "user.email"):
            git(directory, "config", key, git(ROOT, "config", key).stdout.strip())
    remote = git(
        directory,
        "ls-remote",
        "--exit-code",
        "origin",
        f"refs/heads/{BRANCH}",
        check=False,
    )
    if remote.returncode == 0:
        git(directory, "fetch", "origin", BRANCH)
        # Preserve local commits if the last push was interrupted; no force pushes.
        if git(directory, "rev-parse", "--verify", "HEAD", check=False).returncode:
            git(directory, "checkout", "-B", BRANCH, "FETCH_HEAD")
        else:
            git(directory, "merge", "--ff-only", "FETCH_HEAD")
    elif remote.returncode != 2:
        raise RuntimeError(f"Could not inspect data branch: {remote.stderr}")
    return directory


def publish(directory, records, receipts):
    save_feed(directory / "optimized-v1.json", records)
    for path in receipts.glob("*.json"):
        target = directory / "receipts" / path.name
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(path.read_bytes())
    (directory / "README.md").write_text(
        "# optimized-v1 data\n\nVerified paths generated locally. No database runner is used.\n"
        "The website reads optimized-v1.json with bounded caching. Search receipts live in receipts/.\n"
    )
    git(directory, "add", "--", "optimized-v1.json", "README.md")
    if (directory / "receipts").exists():
        git(directory, "add", "--", "receipts")
    if git(directory, "diff", "--cached", "--quiet", check=False).returncode:
        git(directory, "commit", "-m", "Publish verified optimized-v1 solutions")
    git(directory, "push", "origin", f"HEAD:refs/heads/{BRANCH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-manifest", type=Path)
    parser.add_argument("--certificates", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--through", type=date.fromisoformat, default=datetime.now(UTC).date()
    )
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds < 0 or args.max_nodes < 0:
        parser.error("Search budgets must be nonnegative")
    if args.child:
        row = json.loads(args.child.read_text())
        entry = CatalogEntry(
            date.fromisoformat(row["date"]),
            row["name"],
            row["width"],
            row["height"],
            row["puzzle"],
        )
        record, receipt = solve_entry(entry, args.seconds, args.max_nodes)
        atomic_json(
            args.child.with_suffix(".result.json"),
            {"record": record, "receipt": receipt},
        )
        return
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    with (state / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another optimized-v1 worker is active", flush=True)
            return
        publication = publication_checkout(state) if args.publish else None
        output = args.output or state / "optimized-v1.json"
        records = read_feed(output)
        for path in [publication / "optimized-v1.json"] if publication else []:
            for key, row in read_feed(path).items():
                records[key] = choose_record(records.get(key), row)
        if args.bootstrap_manifest:
            for key, row in bootstrap(
                args.bootstrap_manifest, args.certificates
            ).items():
                records[key] = choose_record(records.get(key), row)
        # Never export the unpublished future puzzles present in Google's bundle.
        records = {
            key: row for key, row in records.items() if key <= args.through.isoformat()
        }
        entries = load_entries(args.catalog)
        # A changed official board invalidates its old path, even if solving fails.
        for key, row in list(records.items()):
            entry = entries.get(date.fromisoformat(key))
            if entry and (
                entry.board_sha256 != row["board_sha256"]
                or entry.board() != row["board"]
            ):
                del records[key]
        save_feed(output, records)
        receipts = state / "receipts"
        receipts.mkdir(exist_ok=True)
        failures = []
        for d, entry in sorted(entries.items()):
            if d > args.through or d.isoformat() in records:
                continue
            key = d.isoformat()
            print(f"Solving {key}: {entry.title}", flush=True)
            task = state / "task.json"
            atomic_json(
                task,
                {
                    "date": key,
                    "name": entry.title,
                    "width": entry.width,
                    "height": entry.height,
                    "puzzle": entry.level,
                },
            )
            task.with_suffix(".result.json").unlink(missing_ok=True)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        str(task),
                        "--state-dir",
                        str(state),
                        "--seconds",
                        str(args.seconds),
                        "--max-nodes",
                        str(args.max_nodes),
                    ],
                    check=True,
                    cwd=ROOT,
                    timeout=90 + 2 * args.seconds,
                    env={**os.environ, "PYTHONPATH": str(ROOT)},
                )
                payload = json.loads(task.with_suffix(".result.json").read_text())
                records[key] = choose_record(records.get(key), payload["record"])
                digest = sha256(
                    json.dumps(payload["receipt"], sort_keys=True).encode()
                ).hexdigest()
                atomic_json(receipts / f"{key}-{digest}.json", payload["receipt"])
                save_feed(output, records)
                print(f"Verified {key}: {len(records[key]['moves'])} moves", flush=True)
            except (subprocess.SubprocessError, ValueError) as exc:
                failures.append({"date": key, "error": str(exc)})
                print(f"Unpublished {key}: {exc}", flush=True)
        atomic_json(
            state / "last-run.json",
            {"finished_at": now(), "boards": len(records), "failures": failures},
        )
        if publication:
            publish(publication, records, receipts)
        print(
            json.dumps(
                {
                    "boards": len(records),
                    "failures": failures,
                    "published": bool(publication),
                }
            ),
            flush=True,
        )
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
