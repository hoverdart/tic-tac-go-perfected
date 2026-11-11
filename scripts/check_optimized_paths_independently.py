"""Check shortest-path claims with an independent set-based push A* oracle.

This oracle shares neither bitmask transitions nor the relaxed pattern database
with the optimizer. Wall-only single-O distances provide a weaker lower bound.
An unfinished check is explicitly inconclusive, never an optimality confirmation.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import time
from collections import Counter, deque
from hashlib import sha256
from pathlib import Path

from solver.board_utils import normalize_board
from solver.optimized_paths.bank import load_bank
from solver.optimized_paths.verify_benchmark import verify_cohort
from solver.push_solver.verify import verify_solution

DIRECTIONS = (("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1))
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def check_path(board, moves, *, seconds=1, max_nodes=20_000):
    checked = verify_solution(board, moves)
    if not checked.ok:
        raise ValueError(f"Invalid candidate: {checked.error}")
    started = time.perf_counter()
    deadline = started + seconds
    board = normalize_board(board)
    height, width = len(board), len(board[0])
    floor = {
        r * width + c
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
        if cell != "B"
    }
    os = frozenset(
        r * width + c
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
        if cell == "O"
    )
    xs = frozenset(
        r * width + c
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
        if cell == "X"
    )
    player = next(
        r * width + c
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
        if cell == "U"
    )
    edges = {}
    lines = []
    for cell in sorted(floor):
        r, c = divmod(cell, width)
        for direction, dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and nr * width + nc in floor:
                edges[cell, direction] = nr * width + nc
        for dr, dc in ((0, 1), (1, 0)):
            positions = [(r + i * dr, c + i * dc) for i in range(3)]
            if all(
                rr < height and cc < width and rr * width + cc in floor
                for rr, cc in positions
            ):
                lines.append(frozenset(rr * width + cc for rr, cc in positions))
    distances = {}
    assignments = []
    if len(os) == 2:
        for target in sorted(set().union(*lines)):
            distance = {target: 0}
            queue = deque([target])
            while queue:
                current = queue.popleft()
                for direction, _dr, _dc in DIRECTIONS:
                    previous = edges.get((current, direction))
                    stand = edges.get((previous, direction))
                    if stand is not None and previous not in distance:
                        distance[previous] = distance[current] + 1
                        queue.append(previous)
            distances[target] = distance
        assignments = [
            (a, b)
            for line in lines
            for a in sorted(line)
            for b in sorted(line)
            if a != b
        ]
    h_cache = {}

    def heuristic(pieces):
        if len(pieces) != 2:
            return 0
        if pieces not in h_cache:
            first, second = sorted(pieces)
            h_cache[pieces] = min(
                (
                    distances[a].get(first, math.inf)
                    + distances[b].get(second, math.inf)
                    for a, b in assignments
                ),
                default=math.inf,
            )
        return h_cache[pieces]

    initial_h = heuristic(os)
    if not math.isfinite(initial_h):
        raise AssertionError("Independent wall-only heuristic contradicts valid seed")
    serial = itertools.count()
    start = os, xs, player
    frontier = [(initial_h, 0, next(serial), start, None)]
    best = {start: 0}
    expanded = 0

    def report(status, shorter=None):
        return {
            "status": status,
            "expanded": expanded,
            "discovered_states": len(best),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "claimed_moves": len(moves),
            "shorter_path": shorter,
        }

    while frontier:
        if time.perf_counter() >= deadline:
            return report("inconclusive_timeout")
        f, cost, _order, state, link = heapq.heappop(frontier)
        if best.get(state) != cost:
            continue
        if f >= len(moves):
            return report("confirmed")
        if expanded >= max_nodes:
            return report("inconclusive_node_cap")
        expanded += 1
        os, xs, player = state
        occupied = os | xs
        walks = {player: ""}
        queue = deque([player])
        while queue:
            current = queue.popleft()
            for direction, _dr, _dc in DIRECTIONS:
                target = edges.get((current, direction))
                if (
                    target is not None
                    and target not in occupied
                    and target not in walks
                ):
                    walks[target] = walks[current] + direction
                    queue.append(target)
        finishes = []
        for line in lines:
            missing = line - os
            if not missing:
                finishes.append("")
            elif len(missing) == 1:
                target = next(iter(missing))
                if target in walks:
                    finishes.append(walks[target])
        tail = min(finishes, key=len) if finishes else None
        if tail is not None and cost + len(tail) < len(moves):
            parts = []
            while link is not None:
                link, segment = link
                parts.append(segment)
            shorter = "".join(reversed(parts)) + tail
            if not verify_solution(board, shorter).ok:
                raise AssertionError("Independent oracle reconstructed an invalid path")
            return report("shorter_found", shorter)
        for kind, pieces in (("O", os), ("X", xs)):
            for cell in sorted(pieces):
                for direction, _dr, _dc in DIRECTIONS:
                    stand = edges.get((cell, OPPOSITE[direction]))
                    destination = edges.get((cell, direction))
                    if (
                        stand not in walks
                        or destination is None
                        or destination in occupied
                    ):
                        continue
                    changed = (pieces - {cell}) | {destination}
                    if kind == "X" and any(line <= changed for line in lines):
                        continue
                    child = (
                        (frozenset(changed), xs, cell)
                        if kind == "O"
                        else (os, frozenset(changed), cell)
                    )
                    segment = walks[stand] + direction
                    g = cost + len(segment)
                    if g >= best.get(child, math.inf) or g + heuristic(child[0]) >= len(
                        moves
                    ):
                        continue
                    best[child] = g
                    heapq.heappush(
                        frontier,
                        (
                            g + heuristic(child[0]),
                            g,
                            next(serial),
                            child,
                            (link, segment),
                        ),
                    )
    return report("confirmed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--max-nodes", type=int, default=20_000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    verify_cohort(args.cohort, args.manifest)
    metadata = json.loads((args.cohort / "metadata.json").read_text())
    bank = load_bank(args.manifest)
    rows = [
        json.loads(line)
        for line in (args.cohort / "results.jsonl").read_text().splitlines()
    ]
    selected = [row for row in rows if row.get("result", {}).get("optimality_proven")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (args.output_dir / "results.jsonl").open("w") as output:
        for row in selected:
            result = check_path(
                bank.cases[row["id"]].board,
                row["result"]["moves"],
                seconds=args.seconds,
                max_nodes=args.max_nodes,
            )
            counts[result["status"]] += 1
            output.write(
                json.dumps(
                    {"id": row["id"], "board_sha256": row["board_sha256"], **result}
                )
                + "\n"
            )
            output.flush()
            if result["status"] == "shorter_found":
                raise AssertionError(f"Shortest-path claim contradicted: {row['id']}")
    summary = {
        "scope": "independent_set_based_oracle_not_timing_benchmark",
        "cohort_configuration_sha256": metadata["configuration_sha256"],
        "expected_claims": len(selected),
        "completed_checks": sum(counts.values()),
        "counts": dict(counts),
        "seconds_per_case": args.seconds,
        "max_nodes": args.max_nodes,
        "oracle_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "no_contradictions": not counts["shorter_found"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
