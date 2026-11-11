"""Bank-seeded graph, vicinity, and globally certifying move-cost searches."""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import deque
from dataclasses import asdict, dataclass

from .lower_bound import build_pattern_database
from .model import Budget, Geometry, LimitReached, cells


@dataclass(frozen=True)
class Link:
    previous: Link | None
    moves: str


def reconstruct(link):
    chunks = []
    while link is not None:
        chunks.append(link.moves)
        link = link.previous
    return "".join(reversed(chunks))


@dataclass(frozen=True)
class OptimizationResult:
    moves: str
    baseline_moves: int
    graph_moves: int
    lower_bound: int
    optimality_proven: bool
    termination: str
    expanded: int
    elapsed_ms: float
    verified: bool
    final_board: tuple[tuple[str, ...], ...]
    phases: tuple[dict, ...]

    @property
    def improved(self):
        return len(self.moves) < self.baseline_moves

    def to_dict(self):
        return {
            **asdict(self),
            "improved": self.improved,
            "move_count": len(self.moves),
        }


def observed_graph(geometry, seeds, budget):
    """Shortest path in the union of verified, exact-state trajectories only."""
    best = min(seeds, key=lambda moves: (len(moves), moves))
    edges = {}
    for moves in seeds:
        before = geometry.start
        for move, after in zip(moves, itertools.islice(geometry.trace(moves), 1, None)):
            budget.check()
            edges.setdefault(before, {})[move] = after
            before = after
    parents = {geometry.start: None}
    queue = deque([geometry.start])
    while queue:
        budget.expand()
        state = queue.popleft()
        if geometry.won(state):
            parts = []
            while parents[state] is not None:
                state, move = parents[state]
                parts.append(move)
            candidate = "".join(reversed(parts))
            geometry.verified(candidate)
            return candidate, {
                "states": len(parents),
                "edges": sum(map(len, edges.values())),
            }
        for move, child in edges.get(state, {}).items():
            if child not in parents:
                parents[child] = state, move
                queue.append(child)
    # Every seed is verified. Reaching this branch indicates an implementation bug.
    raise RuntimeError(
        f"Verified seed is absent from observed graph ({len(best)} moves)"
    )


def vicinity_layouts(geometry, seeds, budget, *, radius, max_layouts):
    """Include every anchor, then add bounded geometric push neighborhoods."""
    layouts = set()
    for moves in seeds:
        for state in geometry.trace(moves):
            budget.check()
            layouts.add(state[:2])
            if len(layouts) > max_layouts:
                return None, True
    layer = sorted(layouts)
    for _ in range(radius):
        following = []
        for os, xs in layer:
            budget.check()
            for piece, mask in (("O", os), ("X", xs)):
                for cell in cells(mask):
                    for _move, _stand, dest in geometry.pushes[cell]:
                        if (os | xs) & (1 << dest):
                            continue
                        changed = (mask ^ (1 << cell)) | (1 << dest)
                        layout = (changed, xs) if piece == "O" else (os, changed)
                        if layout in layouts:
                            continue
                        if len(layouts) >= max_layouts:
                            return layouts, True
                        layouts.add(layout)
                        following.append(layout)
        layer = following
    return layouts, False


def bounded_astar(geometry, incumbent, budget, *, pdb=None, layouts=None):
    """Full legal push graph when layouts=None; restricted search otherwise.

    A partially expanded node remains in the lower-bound calculation on every
    interruption. Immutable links prevent reopened states corrupting paths.
    """
    heuristic = pdb.estimate if pdb is not None else lambda _state: 0
    initial_h = heuristic(geometry.start)
    if not math.isfinite(initial_h):
        raise RuntimeError("Verified seed contradicts complete relaxed reachability")
    serial = itertools.count()
    frontier = [(initial_h, 0, next(serial), geometry.start, None)]
    best_cost = {geometry.start: 0}
    best_moves = incumbent
    active_f = None
    reason = "exhausted"
    try:
        while frontier:
            budget.check()
            f, g, _order, state, link = heapq.heappop(frontier)
            if best_cost.get(state) != g:
                continue
            active_f = f
            if f >= len(best_moves):
                # This is the globally cheapest valid frontier label.
                reason = "bound_met"
                active_f = None
                break
            budget.expand()
            walks = geometry.walks(state)
            finish = geometry.finish(state, walks)
            if finish is not None and g + len(finish) < len(best_moves):
                candidate = reconstruct(link) + finish
                geometry.verified(candidate)
                best_moves = candidate
            if f < len(best_moves):
                for child, segment in geometry.successors(state, walks):
                    budget.check()
                    if layouts is not None and child[:2] not in layouts:
                        continue
                    cost = g + len(segment)
                    if cost >= len(best_moves) or cost >= best_cost.get(
                        child, math.inf
                    ):
                        continue
                    h = heuristic(child)
                    if cost + h >= len(best_moves):
                        continue
                    best_cost[child] = cost
                    heapq.heappush(
                        frontier,
                        (cost + h, cost, next(serial), child, Link(link, segment)),
                    )
            active_f = None
    except LimitReached as exc:
        reason = str(exc)
    if reason in ("bound_met", "exhausted"):
        lower_bound = len(best_moves)
    else:
        labels = [
            f for f, g, _order, state, _link in frontier if best_cost.get(state) == g
        ]
        if active_f is not None:
            labels.append(active_f)
        lower_bound = min([len(best_moves), *labels])
    return best_moves, int(lower_bound), reason, len(best_cost)


def optimize(
    board,
    seeds,
    *,
    mode="repair",
    seconds=1.0,
    max_nodes=100_000,
    radius=1,
    max_layouts=8192,
    max_pdb_states=130_000,
):
    """Improve independently verified seeds; no database, network, or fallback.

    `exact` alone may certify nontrivial global optimality. Radius/layout limits
    affect `repair` only. All returned paths pass the independent verifier.
    """
    if mode not in ("graph", "repair", "exact"):
        raise ValueError("Unknown optimization mode")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    for name, value in (
        ("max_nodes", max_nodes),
        ("radius", radius),
        ("max_layouts", max_layouts),
        ("max_pdb_states", max_pdb_states),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    started = time.perf_counter()
    geometry = Geometry(board)
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a collection of move strings")
    seeds = tuple(seeds)
    if not seeds or any(not isinstance(moves, str) for moves in seeds):
        raise ValueError("At least one move-string seed is required")
    seeds = tuple(sorted(set(seeds), key=lambda moves: (len(moves), moves)))
    for moves in seeds:
        geometry.verified(moves)
    incumbent = seeds[0]
    baseline = graph_moves = len(incumbent)
    lower_bound = 0
    budget = Budget(started + seconds, max_nodes)
    phases = []
    reason = "graph_complete"
    phase = "graph"
    try:
        incumbent, stats = observed_graph(geometry, seeds, budget)
        graph_moves = len(incumbent)
        phases.append({"phase": "graph", "moves": graph_moves, **stats})
        if mode != "graph" and incumbent:
            phase = "rewrite"
            budget.check()
            # Reuse the qualified implementation without modifying its sources.
            from solver.push_solver.path_optimizer import PathRewriter
            from solver.push_solver.v4 import Context

            ctx = Context(geometry.board, budget.deadline, None)
            rewritten = PathRewriter(
                ctx, min(budget.deadline, time.perf_counter() + 0.1)
            ).optimize(incumbent, geometry.push_trace(incumbent))
            if len(rewritten.moves) < len(incumbent):
                geometry.verified(rewritten.moves)
                incumbent = rewritten.moves
            phases.append(
                {
                    "phase": "rewrite",
                    "moves": len(incumbent),
                    "evaluations": rewritten.evaluations,
                }
            )
            if mode == "repair":
                phase = "vicinity"
                layouts, capped = vicinity_layouts(
                    geometry,
                    (*seeds, incumbent),
                    budget,
                    radius=radius,
                    max_layouts=max_layouts,
                )
                if layouts is None:
                    reason = "anchor_layout_cap"
                else:
                    incumbent, _restricted_bound, reason, states = bounded_astar(
                        geometry, incumbent, budget, layouts=layouts
                    )
                    phases.append(
                        {
                            "phase": "vicinity",
                            "moves": len(incumbent),
                            "layouts": len(layouts),
                            "layout_cap_hit": capped,
                            "search_states": states,
                            "termination": reason,
                        }
                    )
            else:
                phase = "lower_bound"
                deadline = budget.deadline
                budget.deadline = min(
                    deadline, time.perf_counter() + min(1.0, seconds * 0.25)
                )
                try:
                    pdb = build_pattern_database(
                        geometry, budget, max_states=max_pdb_states
                    )
                finally:
                    budget.deadline = deadline
                lower_bound = pdb.estimate(geometry.start)
                if not math.isfinite(lower_bound):
                    raise RuntimeError("Verified seed contradicts relaxed lower bound")
                phases.append(
                    {
                        "phase": "lower_bound",
                        "complete": pdb.complete,
                        "states": len(pdb.distances),
                        "expanded": pdb.expanded,
                        "termination": pdb.reason,
                        "start_bound": lower_bound,
                    }
                )
                phase = "exact"
                incumbent, lower_bound, reason, states = bounded_astar(
                    geometry, incumbent, budget, pdb=pdb
                )
                phases.append(
                    {
                        "phase": "exact",
                        "moves": len(incumbent),
                        "search_states": states,
                        "termination": reason,
                    }
                )
    except LimitReached as exc:
        reason = str(exc)
        phases.append({"phase": phase, "termination": reason})
    final_board = geometry.verified(incumbent)
    proven = len(incumbent) == lower_bound
    return OptimizationResult(
        incumbent,
        baseline,
        graph_moves,
        int(lower_bound),
        proven,
        reason,
        budget.expanded,
        (time.perf_counter() - started) * 1000,
        True,
        final_board,
        tuple(phases),
    )
