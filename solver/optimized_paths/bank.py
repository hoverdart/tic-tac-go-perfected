"""Fingerprint and independently verify explicit archive inputs, never dates alone."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver.verify import verify_solution

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    id: str
    name: str
    board_sha256: str
    board: tuple[tuple[str, ...], ...]


@dataclass
class SolutionBank:
    cases: dict[str, Case]
    paths: dict[str, dict[str, list[dict]]]
    manifest: dict
    counts: dict
    rejected: list[dict]


def load_bank(manifest_path, *, root=ROOT):
    manifest_path = Path(manifest_path)
    raw_manifest = manifest_path.read_bytes()
    spec = json.loads(raw_manifest)
    if (
        not isinstance(spec, dict)
        or spec.get("schema") != "optimized-path-inputs-v1"
        or set(spec) != {"schema", "catalog", "sources"}
        or not isinstance(spec["catalog"], str)
        or not isinstance(spec["sources"], list)
        or not all(isinstance(p, str) for p in spec["sources"])
    ):
        raise ValueError("Unsupported bank manifest")
    if len(set(spec["sources"])) != len(spec["sources"]):
        raise ValueError("Duplicate archive sources")
    root = Path(root)
    catalog_bytes = (root / spec["catalog"]).read_bytes()
    cases = {}
    for line in catalog_bytes.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        width, height, puzzle = row["width"], row["height"], row["puzzle"]
        if (
            type(width) is not int
            or type(height) is not int
            or not 3 <= width <= 8
            or not 3 <= height <= 8
            or len(puzzle) != width * height
            or set(puzzle) - set("WPOX-")
            or puzzle.count("P") != 1
        ):
            raise ValueError("Invalid catalog geometry")
        entry = CatalogEntry(
            date.fromisoformat(row["date"]), row["name"], width, height, puzzle
        )
        key = entry.puzzle_date.strftime("%Y%m%d")
        if (
            key != row["id"]
            or key in cases
            or row["board_sha256"] != entry.board_sha256
        ):
            raise ValueError("Invalid or duplicate catalog identity")
        cases[key] = Case(
            key, entry.title, entry.board_sha256, tuple(map(tuple, entry.board()))
        )
    if not cases:
        raise ValueError("Empty catalog")
    paths = {key: {} for key in cases}
    counts = Counter()
    rejected = []
    sources = []
    verified = {}
    for name in spec["sources"]:
        content = (root / name).read_bytes()
        digest = sha256(content).hexdigest()
        sources.append({"path": name, "sha256": digest})
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an archive object at {name}:{number}")
            counts["rows_read"] += 1
            key = str(row.get("id") or row.get("date", "").replace("-", ""))
            reason = None
            if key not in cases:
                reason = "unknown_board"
            elif row.get("transformation"):
                reason = "transformed_coordinates"
            elif row.get("board_sha256") != cases[key].board_sha256:
                reason = "board_hash_mismatch"
            result = row.get("result")
            moves = row.get("solution")
            if moves is None and isinstance(result, dict):
                moves = result.get("moves")
            if reason is None and not isinstance(moves, str):
                counts["rows_without_path"] += 1
                continue
            if reason is None:
                counts["candidate_rows"] += 1
                identity = key, moves
                if identity not in verified:
                    verified[identity] = verify_solution(cases[key].board, moves).ok
                if not verified[identity]:
                    reason = "invalid_path"
            if reason is not None:
                counts[reason] += 1
                rejected.append(
                    {"source": name, "line": number, "id": key, "reason": reason}
                )
                continue
            paths[key].setdefault(moves, []).append(
                {"source": name, "line": number, "source_sha256": digest}
            )
    counts["unique_paths_replayed"] = len(verified)
    counts["unique_valid_paths"] = sum(map(len, paths.values()))
    counts["boards_with_seeds"] = sum(bool(rows) for rows in paths.values())
    counts["boards_without_seeds"] = len(cases) - counts["boards_with_seeds"]
    provenance = {
        "schema": spec["schema"],
        "manifest_sha256": sha256(raw_manifest).hexdigest(),
        "catalog": {
            "path": spec["catalog"],
            "sha256": sha256(catalog_bytes).hexdigest(),
        },
        "sources": sources,
    }
    return SolutionBank(cases, paths, provenance, dict(counts), rejected)
