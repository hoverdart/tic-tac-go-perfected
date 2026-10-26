"""Cooperative, bitmask push search. V3 and its verifier remain independent.

All geometry and reachability caches belong to one invocation. Searches share
successors, never their objective-specific dominance tables. Yielding generators
retain partial work at every scheduling boundary, including reverse BFS queries.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import heapq
import itertools
import math
import platform
from pathlib import Path
import resource
import time

from solver.board_utils import normalize_board
from solver.push_solver.models import DIRECTIONS, Push, PushSolveResult, SearchAttempt
from solver.push_solver.verify import verify_solution

StateKey = tuple[int, int, int]  # O occupancy, X occupancy, player-reachable region
SCHEDULES = {
    "rank": (4, 2, 1, 1),
    "uniform": (1, 1, 1, 1),
    "weighted": (2, 4, 1, 1),
    "beam": (1, 1, 1, 4),
    "beam8": (1, 1, 1, 8),
}
STRATEGIES = ("rank_discrepancy", "weighted", "macro", "committed_beam")


def cells(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def mask_of(indices):
    return sum(1 << i for i in indices)


class BudgetExpired(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Plan:
    score: float
    line: int
    target: int
    assignment: tuple[tuple[int, int], tuple[int, int]]
    route: int
    stands: int


@dataclass(slots=True)
class ScoreEntry:
    h: float
    geometry: tuple
    rows: tuple
    plans: tuple[Plan, ...] = ()


@dataclass(frozen=True, slots=True)
class Child:
    pushes: tuple[Push, ...]
    state: StateKey
    h: float
    bias: float
    policy: float = 0.0


@dataclass(frozen=True, slots=True)
class Link:
    previous: Link | None
    pushes: tuple[Push, ...]


@dataclass(frozen=True, slots=True)
class Found:
    state: StateKey
    target: int | None
    link: Link | None


class Context:
    def __init__(self, board, deadline, max_nodes, *, route_limit=3):
        self.started = time.perf_counter()
        self.deadline = deadline
        self.max_nodes = max_nodes
        self.board = normalize_board(board)
        self.rows, self.cols = len(self.board), len(self.board[0])
        self.size = self.rows * self.cols
        self.all = (1 << self.size) - 1
        self.walls = mask_of(
            i
            for i in range(self.size)
            if self.board[i // self.cols][i % self.cols] == "B"
        )
        self.floor = self.all ^ self.walls
        self.os = mask_of(
            i
            for i in range(self.size)
            if self.board[i // self.cols][i % self.cols] == "O"
        )
        self.xs = mask_of(
            i
            for i in range(self.size)
            if self.board[i // self.cols][i % self.cols] == "X"
        )
        self.player = next(
            i
            for i in range(self.size)
            if self.board[i // self.cols][i % self.cols] == "U"
        )
        self.adj = [[] for _ in range(self.size)]
        self.transitions = [[] for _ in range(self.size)]
        self.reverse = [[] for _ in range(self.size)]
        self.push_lookup = {}
        for i in cells(self.floor):
            r, c = divmod(i, self.cols)
            for move, dr, dc in DIRECTIONS:
                nr, nc, sr, sc = r + dr, c + dc, r - dr, c - dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    dest = nr * self.cols + nc
                    if self.floor & (1 << dest):
                        self.adj[i].append((move, dest))
                        if 0 <= sr < self.rows and 0 <= sc < self.cols:
                            stand = sr * self.cols + sc
                            if self.floor & (1 << stand):
                                self.transitions[i].append((move, stand, dest))
                                self.reverse[dest].append((i, stand))
                                self.push_lookup[i, move] = stand, dest
        self.adj_masks = [mask_of(j for _, j in row) for row in self.adj]
        # Match the reference's U/D/L/R reverse-BFS tie order. Canonical route
        # ties affect successor ranks, even though they do not affect legality.
        for dest, edges in enumerate(self.reverse):
            edges.sort(
                key=lambda edge: (
                    0
                    if edge[0] // self.cols > dest // self.cols
                    else 1
                    if edge[0] // self.cols < dest // self.cols
                    else 2
                    if edge[0] > dest
                    else 3
                )
            )
        self.lines = []
        for r in range(self.rows):
            for c in range(self.cols - 2):
                line = mask_of(r * self.cols + c + k for k in range(3))
                if not line & self.walls:
                    self.lines.append(line)
        for r in range(self.rows - 2):
            for c in range(self.cols):
                line = mask_of((r + k) * self.cols + c for k in range(3))
                if not line & self.walls:
                    self.lines.append(line)
        self.lines_at = [
            tuple(line for line in self.lines if line & (1 << i))
            for i in range(self.size)
        ]
        self.start = (self.os, self.xs, self.reachable(self.player, self.os | self.xs))
        self.distances = {}
        self.next_cells = {}
        self.route_cache = OrderedDict()
        self.reverse_cache = OrderedDict()
        self.frozen_cache = OrderedDict()
        self.deadlock_cache = OrderedDict()
        self.edge_cache = OrderedDict()
        self.geometry_cache = OrderedDict()
        self.cost_geometry_cache = OrderedDict()
        self.goal_cache = OrderedDict()
        self.score_cache = OrderedDict()
        self.successor_cache = OrderedDict()
        self.metrics = Counter()
        self.expanded = 0
        self.route_limit = route_limit
        self.peak_frontier = 0
        self.preprocess_ms = 0.0
        self.ranker = None
        self.beam_mode = "interleaved"
        self.macro_mode = "straight"

    def check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise BudgetExpired("timeout")

    def checkpoint(self):
        self.check()
        yield None

    @staticmethod
    def put(cache, key, value, limit=8192):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)
        return value

    def reachable(self, start, occupied):
        allowed = self.floor & ~occupied
        seen = frontier = (1 << start) & allowed
        while frontier:
            neighbors = 0
            for i in cells(frontier):
                neighbors |= self.adj_masks[i]
            frontier = neighbors & allowed & ~seen
            seen |= frontier
        return seen

    def initialize(self):
        for target in sorted(set(i for line in self.lines for i in cells(line))):
            yield from self.checkpoint()
            distances = [-1] * self.size
            distances[target] = 0
            nexts = [-1] * self.size
            queue = [target]
            for current in queue:
                for previous, _stand in self.reverse[current]:
                    if distances[previous] < 0:
                        distances[previous] = distances[current] + 1
                        nexts[previous] = current
                        queue.append(previous)
            self.distances[target] = distances
            self.next_cells[target] = nexts
        self.preprocess_ms = (time.perf_counter() - self.started) * 1000

    def goal(self, state):
        os, xs, region = state
        if os not in self.goal_cache:
            targets = []
            for line in self.lines:
                count = (os & line).bit_count()
                if count == 3:
                    targets.append(None)
                elif count == 2:
                    targets.append((line & ~os).bit_length() - 1)
            self.put(self.goal_cache, os, tuple(targets), 256)
        for target in self.goal_cache[os]:
            if target is None or region & (1 << target):
                return True, target
        return False, None

    def x_loss(self, xs, dest=None):
        return any(
            xs & line == line
            for line in (self.lines if dest is None else self.lines_at[dest])
        )

    def reverse_map(self, blockers, target):
        self.metrics["reverse_queries"] += 1
        key = blockers, target
        if key in self.reverse_cache:
            self.metrics["reverse_hits"] += 1
            self.reverse_cache.move_to_end(key)
            return self.reverse_cache[key]
        self.metrics["reverse_computations"] += 1
        seen = 0 if blockers & (1 << target) else (1 << target)
        queue = list(cells(seen))
        for current in queue:
            yield from self.checkpoint()
            for previous, stand in self.reverse[current]:
                bit = 1 << previous
                if not (seen & bit or blockers & (bit | (1 << stand))):
                    seen |= bit
                    queue.append(previous)
        return self.put(self.reverse_cache, key, seen)

    def frozen(self, occupied):
        if occupied in self.frozen_cache:
            self.metrics["frozen_hits"] += 1
            return self.frozen_cache[occupied]
        frozen = 0
        while True:
            before = frozen
            blockers = self.walls | frozen
            for i in cells(occupied & ~frozen):
                if all(
                    blockers & ((1 << stand) | (1 << dest))
                    for _, stand, dest in self.transitions[i]
                ):
                    frozen |= 1 << i
            if before == frozen:
                break
        return self.put(self.frozen_cache, occupied, frozen, 16384)

    def deadlock(self, state):
        self.metrics["deadlock_queries"] += 1
        os, xs, region = state
        if os.bit_count() < 2:
            return True
        frozen = self.frozen(os | xs)
        if not frozen:
            return os.bit_count() == 2 and not (yield from self.geometry(os))
        start = (region & -region).bit_length() - 1
        accessible = self.reachable(start, frozen)
        key = os, frozen, accessible
        if key in self.deadlock_cache:
            self.metrics["deadlock_hits"] += 1
            self.deadlock_cache.move_to_end(key)
            return self.deadlock_cache[key]
        result = yield from self.frozen_deadlock(os, frozen, accessible)
        return self.put(self.deadlock_cache, key, result, 8192)

    def frozen_deadlock(self, os, frozen, accessible):
        frozen_os, frozen_xs = os & frozen, frozen & ~os
        blockers = self.walls | frozen
        for line in self.lines:
            yield from self.checkpoint()
            if frozen_xs & line:
                continue
            if os.bit_count() >= 3:
                # OOO wins without requiring the player to enter the line.
                candidates = []
                for target in cells(line):
                    reverse = yield from self.reverse_map(blockers, target)
                    candidates.append((os & reverse) | (frozen_os & (1 << target)))
                if any(
                    a != b and c & ~((1 << a) | (1 << b))
                    for a in cells(candidates[0])
                    for b in cells(candidates[1])
                    for c in (candidates[2],)
                ):
                    return False
            # With >2 O pieces, unused frozen O pieces may stay outside the line.
            if os.bit_count() == 2 and frozen_os & ~line:
                continue
            for target in cells(line & ~blockers & accessible):
                a, b = tuple(cells(line & ~(1 << target)))
                ra = yield from self.reverse_map(blockers, a)
                rb = yield from self.reverse_map(blockers, b)
                ca = (os & ra) | (frozen_os & (1 << a))
                cb = (os & rb) | (frozen_os & (1 << b))
                if ca and cb and (ca | cb).bit_count() >= 2:
                    return False
        return True

    def canonical_route(self, source, target):
        distance = self.distances[target][source]
        if distance < 0:
            return None
        route = 1 << source
        stands = 0
        current = source
        while current != target:
            nxt = self.next_cells[target][current]
            stand = 2 * current - nxt
            stands |= 1 << stand
            route |= (1 << nxt) | (1 << stand)
            current = nxt
        return distance, route, stands

    def routes(self, source, target):
        key = source, target
        if key in self.route_cache:
            self.metrics["route_hits"] += 1
            return self.route_cache[key]
        canonical = self.canonical_route(source, target)
        if canonical is None:
            return ()
        found = [canonical]
        if self.route_limit > 1 and source != target:
            shortest = canonical[0]
            # A* enumerates simple routes, favoring depth on equal f to avoid
            # expanding the combinatorial breadth of open-board shortest paths.
            queue = [(shortest, 0, (source,), 1 << source, 0)]
            popped = 0
            while queue and len(found) < self.route_limit and popped < 4096:
                _, neg_depth, path, used, stands = heapq.heappop(queue)
                popped += 1
                if popped % 16 == 0:
                    yield from self.checkpoint()
                current = path[-1]
                depth = -neg_depth
                if current == target:
                    route = used | stands
                    candidate = depth, route, stands
                    if candidate not in found:
                        found.append(candidate)
                    continue
                for _move, stand, dest in self.transitions[current]:
                    remaining = self.distances[target][dest]
                    if (
                        used & (1 << dest)
                        or remaining < 0
                        or depth + 1 + remaining > shortest + 2
                    ):
                        continue
                    heapq.heappush(
                        queue,
                        (
                            depth + 1 + remaining,
                            -depth - 1,
                            path + (dest,),
                            used | (1 << dest),
                            stands | (1 << stand),
                        ),
                    )
        self.metrics["route_computations"] += 1
        return self.put(self.route_cache, key, tuple(found))

    def geometry(self, os):
        if os in self.geometry_cache:
            self.metrics["geometry_hits"] += 1
            self.geometry_cache.move_to_end(os)
            return self.geometry_cache[os]
        result = []
        for line in self.lines:
            yield from self.checkpoint()
            for target in cells(line):
                a, b = tuple(cells(line & ~(1 << target)))
                o_cells = (
                    tuple(frozenset(cells(os)))
                    if os.bit_count() == 2
                    else tuple(cells(os))
                )
                for oa, ob in itertools.combinations(o_cells, 2):
                    for ta, tb in ((a, b), (b, a)):
                        route_a = self.canonical_route(oa, ta)
                        route_b = self.canonical_route(ob, tb)
                        if route_a is not None and route_b is not None:
                            result.append(
                                (line, target, oa, ta, ob, tb, route_a, route_b)
                            )
        self.metrics["geometry_computations"] += 1
        return self.put(self.geometry_cache, os, tuple(result), 256)

    @staticmethod
    def materialize_plan(geometry, row):
        score, index, route_a, route_b = row
        line, target, oa, a, ob, b, _, _ = geometry[index]
        return Plan(
            score,
            line,
            target,
            ((oa, a), (ob, b)),
            route_a[1] | route_b[1],
            route_a[2] | route_b[2],
        )

    def ranked_rows(self, state, *, full=False):
        os, xs, _ = state
        geometry = yield from self.geometry(os)
        compiled = self.cost_geometry_cache.get(os)
        if compiled is None:
            compiled = tuple(
                (
                    row[0],
                    1 << row[1],
                    row[6][0]
                    + row[7][0]
                    + bool(row[6][1] & (1 << row[4]))
                    + bool(row[7][1] & (1 << row[2])),
                    row[6][1],
                    row[7][1],
                )
                for row in geometry
            )
            compiled = (
                compiled,
                tuple({row[0] for row in compiled}),
                tuple({mask for row in compiled for mask in row[3:]}),
            )
            self.put(self.cost_geometry_cache, os, compiled, 256)
        else:
            self.cost_geometry_cache.move_to_end(os)
        compiled, lines, masks = compiled
        line_costs = {line: 0.5 * (xs & line).bit_count() for line in lines}
        route_counts = {mask: (xs & mask).bit_count() for mask in masks}
        candidates = []
        for index, (line, target_bit, base, route_mask_a, route_mask_b) in enumerate(
            compiled
        ):
            if index % 32 == 0:
                yield from self.checkpoint()
            score = (
                base
                + route_counts[route_mask_a]
                + route_counts[route_mask_b]
                + line_costs[line]
                + (2.0 if xs & target_bit else 0.0)
            )
            candidates.append((score, index, geometry[index][6], geometry[index][7]))
        candidates.sort()
        # Refine only promising obstructed candidates; static route alternatives
        # are cached and remain ordering evidence, never a pruning condition.
        refined = []
        choices = {}
        for score, index, route_a, route_b in candidates if full else candidates[:24]:
            line, target, oa, a, ob, b, _, _ = geometry[index]
            if self.route_limit > 1 and xs & (route_a[1] | route_b[1]):
                for source, route_target, other in ((oa, a, ob), (ob, b, oa)):
                    route_key = source, route_target, other
                    if route_key not in choices:
                        alternatives = yield from self.routes(source, route_target)
                        choices[route_key] = min(
                            alternatives,
                            key=lambda r: (
                                r[0]
                                + (xs & r[1]).bit_count()
                                + bool(r[1] & (1 << other))
                            ),
                        )
                route_a, route_b = choices[oa, a, ob], choices[ob, b, oa]
                score = (
                    route_a[0]
                    + route_b[0]
                    + (xs & route_a[1]).bit_count()
                    + (xs & route_b[1]).bit_count()
                    + bool(route_a[1] & (1 << ob))
                    + bool(route_b[1] & (1 << oa))
                    + 0.5 * (xs & line).bit_count()
                    + (2.0 if xs & (1 << target) else 0)
                )
            refined.append((score, index, route_a, route_b))
        refined.sort()
        return geometry, tuple(refined)

    def score(self, state, limit=24):
        """Cache scored rows once; allocate only the requested plan prefix.

        The current component determines terminal reachability, while all
        nonterminal plan scores depend only on O/X occupancy. A cached result
        with fewer than `limit` plans is still complete, including an empty one.
        """
        if not 1 <= limit <= 24:
            raise ValueError("Plan prefix must be between 1 and 24")
        if self.goal(state)[0]:
            return 0.0, ()
        key = state[:2]
        entry = self.score_cache.get(key)
        if entry is None:
            geometry, rows = yield from self.ranked_rows(state)
            entry = ScoreEntry(rows[0][0] if rows else math.inf, geometry, rows)
            self.put(self.score_cache, key, entry, 8192)
            self.metrics["score_computations"] += 1
        else:
            self.metrics["score_hits"] += 1
            self.score_cache.move_to_end(key)
        wanted = min(limit, len(entry.rows))
        while len(entry.plans) < wanted:
            yield from self.checkpoint()
            # Another session can have extended the same prefix at this yield.
            index = len(entry.plans)
            if index >= wanted:
                break
            entry.plans += (self.materialize_plan(entry.geometry, entry.rows[index]),)
            self.metrics["plans_materialized"] += 1
        return entry.h, entry.plans[:limit]

    def beam_plans(self, state):
        """Cover distinct lines, then player targets, then O assignments.

        Full refinement is confined to the initial commitment pool; ordinary
        successors retain their reference-compatible top-24 scoring semantics.
        No winning line is removed by the ordinary score-cache prefix limit.
        """
        geometry, rows = yield from self.ranked_rows(state, full=True)
        os, xs, region = state
        frozen = self.frozen(os | xs)
        frozen_os, frozen_xs = os & frozen, xs & frozen
        accessible = self.reachable(next(cells(region)), frozen)
        blockers = self.walls | frozen
        reaches = {}
        plans = []
        for row in rows:
            yield from self.checkpoint()
            plan = self.materialize_plan(geometry, row)
            if plan.line & frozen_xs:
                continue
            if os.bit_count() == 2 and not accessible & (1 << plan.target):
                continue
            viable = True
            for source, target in plan.assignment:
                if source == target:
                    continue
                if frozen_os & (1 << source):
                    viable = False
                    break
                if target not in reaches:
                    reaches[target] = yield from self.reverse_map(blockers, target)
                if not reaches[target] & (1 << source):
                    viable = False
                    break
            if viable:
                plans.append(plan)
        self.metrics["pruned_impossible_commitments"] += len(rows) - len(plans)
        self.initial_anchor_plans = tuple(plans[:24])
        ordered = []
        used = set()
        lines = set()
        targets = set()
        for index, plan in enumerate(plans):
            yield from self.checkpoint()
            if plan.line not in lines:
                lines.add(plan.line)
                targets.add((plan.line, plan.target))
                used.add(index)
                ordered.append(plan)
        for index, plan in enumerate(plans):
            if (plan.line, plan.target) not in targets:
                targets.add((plan.line, plan.target))
                used.add(index)
                ordered.append(plan)
        ordered.extend(plan for index, plan in enumerate(plans) if index not in used)
        self.metrics["commitment_plans"] = len(ordered)
        self.metrics["commitment_lines"] = len(lines)
        return tuple(ordered)

    def o_push_count(self, state):
        os, xs, region = state
        occupied = os | xs
        return sum(
            bool(region & (1 << stand)) and not occupied & (1 << dest)
            for i in cells(os)
            for _, stand, dest in self.transitions[i]
        )

    def plan_bias(self, plan, push, dest, before, after):
        bit, dbit = 1 << push.cell, 1 << dest
        before_region, after_region = before[2], after[2]
        bias = 0.0
        if push.piece == "X":
            if push.cell == plan.target:
                bias -= 4.0
            elif bit & plan.line:
                bias -= 2.5
            elif bit & plan.route:
                bias -= 2.0
            elif bit & plan.stands:
                bias -= 1.5
            if dest == plan.target:
                bias += 4.0
            elif dbit & plan.line:
                bias += 2.5
            elif dbit & plan.route:
                bias += 2.0
            elif dbit & plan.stands:
                bias += 1.5
        else:
            assigned = dict(plan.assignment).get(push.cell)
            if assigned is not None:
                old, new = (
                    self.distances[assigned][push.cell],
                    self.distances[assigned][dest],
                )
                old = math.inf if old < 0 else old
                new = math.inf if new < 0 else new
                if new < old:
                    bias -= min(old - new, 3) * 0.5
                elif new > old:
                    bias += min(new - old, 3) * 0.35
                if dest == assigned:
                    bias -= 2.0
            elif bit & (plan.line | plan.route | plan.stands):
                bias += 0.4
        target_bit = 1 << plan.target
        if not target_bit & before_region and target_bit & after_region:
            bias -= 7.0
        elif not target_bit & after_region:
            bias += 0.5
        return bias

    def _produce_successors(self, state, only=None):
        os, xs, region = state
        parent_h, parent_plans = yield from self.score(state, limit=8)
        occupied = os | xs
        parent_o_count = self.o_push_count(state)
        for piece, pieces in (("O", os), ("X", xs)):
            for cell in cells(pieces):
                for move, stand, dest in self.transitions[cell]:
                    if only is not None and (piece, cell, move) != (
                        only.piece,
                        only.cell,
                        only.move,
                    ):
                        continue
                    key = state, piece, cell, move
                    yield from self.checkpoint()
                    if key in self.edge_cache:
                        self.metrics["edge_hits"] += 1
                        self.edge_cache.move_to_end(key)
                        cached = self.edge_cache[key]
                        if cached is not None:
                            yield cached
                        continue
                    self.metrics["generated"] += 1
                    if not region & (1 << stand) or occupied & (1 << dest):
                        self.metrics["pruned_illegal"] += 1
                        self.put(self.edge_cache, key, None, 32768)
                        continue
                    new_os, new_xs = os, xs
                    if piece == "O":
                        new_os = (os ^ (1 << cell)) | (1 << dest)
                    else:
                        new_xs = (xs ^ (1 << cell)) | (1 << dest)
                        if self.x_loss(new_xs, dest):
                            self.metrics["pruned_x_loss"] += 1
                            self.put(self.edge_cache, key, None, 32768)
                            continue
                    next_region = self.reachable(cell, new_os | new_xs)
                    child_state = new_os, new_xs, next_region
                    if not self.goal(child_state)[0] and (
                        yield from self.deadlock(child_state)
                    ):
                        self.metrics["pruned_deadlock"] += 1
                        self.put(self.edge_cache, key, None, 32768)
                        continue
                    h, plans = yield from self.score(child_state, limit=1)
                    push = Push(piece, cell, move)
                    bias = 0.35 if piece == "X" else -0.1
                    if h < parent_h:
                        bias -= 1.0
                    elif h > parent_h:
                        bias += 0.25
                    bias -= (
                        min(max(0, next_region.bit_count() - region.bit_count()), 10)
                        * 1.8
                    )
                    bias -= (
                        min(max(0, self.o_push_count(child_state) - parent_o_count), 4)
                        * 0.8
                    )
                    if parent_plans and piece == "X":
                        plan = parent_plans[0]
                        if cell == plan.target:
                            bias -= 2.0
                        elif (1 << cell) & (plan.line | plan.route):
                            bias -= 1.0
                        if dest == plan.target:
                            bias += 2.0
                        elif (1 << dest) & (plan.line | plan.route):
                            bias += 1.0
                    if parent_plans and plans:
                        parent, child = parent_plans[0], plans[0]
                        if not next_region & (1 << child.target):
                            bias += 2.0
                        if xs & (1 << parent.target) and not new_xs & (
                            1 << child.target
                        ):
                            bias -= 2.0
                        if not region & (1 << parent.target) and next_region & (
                            1 << child.target
                        ):
                            bias -= 11.5
                    if parent_plans:
                        bias += min(
                            self.plan_bias(p, push, dest, state, child_state)
                            for p in parent_plans[:8]
                        )
                    self.metrics["generated_states"] += 1
                    child = Child((push,), child_state, h, bias)
                    self.put(self.edge_cache, key, child, 32768)
                    yield child

    def single_successor(self, state, push):
        """Extend a macro without generating unrelated pushes and their scores."""
        child = None
        for item in self._produce_successors(state, only=push):
            if item is None:
                yield None
            else:
                child = item
        return child

    def macro_continuation(self, state, push):
        if self.macro_mode == "straight":
            return (yield from self.single_successor(state, push))
        legal = []
        for move, _, _ in self.transitions[push.cell]:
            child = yield from self.single_successor(
                state, Push(push.piece, push.cell, move)
            )
            if child is not None:
                legal.append(child)
                if len(legal) > 1:
                    return None
        if len(legal) == 1 and legal[0].pushes[0].move == push.move:
            return legal[0]
        return None

    def commitment_bias(self, plan, parent, child, parent_h):
        """V3 commitment objective; independent from primitive move ordering."""
        important = plan.line | plan.route | plan.stands | (1 << plan.target)
        before, after = parent[2], child.state[2]
        bias = 2.0 * (
            (child.state[1] & important).bit_count()
            - (parent[1] & important).bit_count()
        )
        target = 1 << plan.target
        if not before & target and after & target:
            bias -= 5.0
        elif before & target and not after & target:
            bias += 2.0
        gain = self.o_push_count(child.state) - self.o_push_count(parent)
        if gain > 0:
            bias -= min(gain, 4) * 0.75
        if child.h < parent_h:
            bias -= min(parent_h - child.h, 4.0) * 0.35
        push = child.pushes[0]
        dest = self.push_lookup[push.cell, push.move][1]
        if push.piece == "X":
            if important & (1 << push.cell):
                bias -= 2.0
            if important & (1 << dest):
                bias += 2.5
        else:
            deltas = [
                self.distances[t][dest] - self.distances[t][push.cell]
                for t in cells(plan.line & ~target)
                if self.distances[t][dest] >= 0 and self.distances[t][push.cell] >= 0
            ]
            if deltas:
                delta = min(deltas)
                bias += delta * 1.5 if delta < 0 else min(delta, 3) * 0.75
            elif important & (1 << push.cell):
                bias += 1.0
        if dest == plan.target:
            bias += 4.0
        elif push.cell == plan.target:
            bias -= 3.0
        return bias

    def successors(self, state):
        # Consumers retain their own index into one shared, progressive producer.
        # A scheduling yield never publishes an incomplete list as complete.
        if state not in self.successor_cache:
            if len(self.successor_cache) >= 8192:
                # Never evict partial producers: other sessions may be suspended
                # inside them. Completed entries can be safely recomputed.
                for key, value in self.successor_cache.items():
                    if value[2]:
                        del self.successor_cache[key]
                        self.metrics["successor_evictions"] += 1
                        break
            self.successor_cache[state] = [[], self._produce_successors(state), False]
            self.metrics["successor_computations"] += 1
        else:
            self.metrics["successor_hits"] += 1
            self.successor_cache.move_to_end(state)
        entry = self.successor_cache[state]
        index = 0
        while True:
            if index < len(entry[0]):
                yield entry[0][index]
                index += 1
            elif entry[2]:
                return
            else:
                try:
                    item = next(entry[1])
                except StopIteration:
                    entry[2] = True
                    entry[1] = None
                    continue
                if item is None:
                    yield None
                else:
                    entry[0].append(item)

    def children(self, state, macros=False):
        items = []
        for child in self.successors(state):
            if child is None:
                yield None
            else:
                items.append(child)
        if macros:
            for child in tuple(items):
                chain = child.pushes
                current = child.state
                cell = self.push_lookup[chain[0].cell, chain[0].move][1]
                bias = child.bias
                for _ in range(4):
                    continuation = yield from self.macro_continuation(
                        current, Push(chain[0].piece, cell, chain[0].move)
                    )
                    if continuation is None:
                        break
                    chain += continuation.pushes
                    current = continuation.state
                    cell = self.push_lookup[cell, chain[0].move][1]
                    bias += continuation.bias
                if len(chain) > 1:
                    h, _ = yield from self.score(current, limit=1)
                    items.append(
                        Child(
                            chain,
                            current,
                            h,
                            bias / len(chain) - 0.2 * (len(chain) - 1),
                        )
                    )
        items.sort(key=lambda c: (c.h + c.bias, c.h))
        if self.ranker is not None:
            parent_h, _ = yield from self.score(state, limit=1)
            ranked = []
            for rank, child in enumerate(items, 1):
                yield from self.checkpoint()
                policy_score = self.ranker.score(
                    self, state, child, rank, len(items), parent_h
                )
                ranked.append(
                    (
                        child.h + child.bias - policy_score,
                        rank,
                        Child(
                            child.pushes, child.state, child.h, child.bias, policy_score
                        ),
                    )
                )
            ranked.sort(key=lambda row: (row[0], row[1]))
            items = [row[2] for row in ranked]
        return items

    def walk(self, start, target, occupied):
        if start == target:
            return ""
        paths = {start: ""}
        queue = [start]
        for current in queue:
            for move, nxt in self.adj[current]:
                if occupied & (1 << nxt) or nxt in paths:
                    continue
                paths[nxt] = paths[current] + move
                if nxt == target:
                    return paths[nxt]
                queue.append(nxt)
        raise RuntimeError("V4 reconstruction could not reach a stand cell")

    def reconstruct(self, found, deadline=None):
        chunks = []
        link = found.link
        while link is not None:
            if deadline is not None and time.perf_counter() >= deadline:
                raise BudgetExpired("reconstruction_timeout")
            chunks.append(link.pushes)
            link = link.previous
        pushes = tuple(p for chunk in reversed(chunks) for p in chunk)
        player, os, xs = self.player, self.os, self.xs
        moves = []
        for push in pushes:
            if deadline is not None and time.perf_counter() >= deadline:
                raise BudgetExpired("reconstruction_timeout")
            stand, dest = self.push_lookup[push.cell, push.move]
            moves.append(self.walk(player, stand, os | xs))
            moves.append(push.move)
            if push.piece == "O":
                os = (os ^ (1 << push.cell)) | (1 << dest)
            else:
                xs = (xs ^ (1 << push.cell)) | (1 << dest)
            player = push.cell
        if found.target is not None:
            moves.append(self.walk(player, found.target, os | xs))
        return "".join(moves), pushes


class Session:
    def __init__(self, context, kind, *, weight=2.0, plan=None):
        self.context, self.kind, self.weight, self.plan = context, kind, weight, plan
        self.nodes = 0
        self.peak = 1
        self.frontier_size = 1
        self.frontier_peak = 1
        self.elapsed = 0.0
        self.cpu = 0.0
        self.counters = Counter()
        self.done = False
        self.found = None
        self.iterator = self.search()
        self.counter = itertools.count()
        self.goals = []
        self.anchor_goals = []
        self.slice_node_limit = None
        self.slice_deadline = None

    def expand(self):
        ctx = self.context
        ctx.check()
        if ctx.max_nodes is not None and ctx.expanded >= ctx.max_nodes:
            raise BudgetExpired("node_cap")
        ctx.expanded += 1
        self.nodes += 1

    def search(self):
        if self.kind == "committed_beam":
            yield from self.beam_search()
            return
        ctx = self.context
        h, _ = yield from ctx.score(ctx.start, limit=1)
        queue = [(self.weight * h, 0, 0, next(self.counter), ctx.start, None)]
        best = {ctx.start: (0, 0)}
        while queue:
            yield from ctx.checkpoint()
            _, rank, cost, _, current, link = heapq.heappop(queue)
            key = (rank, cost) if self.kind == "rank_discrepancy" else (0, cost)
            if best.get(current) != key:
                continue
            self.expand()
            won, target = ctx.goal(current)
            if won:
                yield Found(current, target, link)
                return
            children = yield from ctx.children(current, self.kind == "macro")
            for local_rank, child in enumerate(children):
                yield from ctx.checkpoint()
                g = cost + len(child.pushes)
                discrepancy = (
                    rank + local_rank if self.kind == "rank_discrepancy" else 0
                )
                metric = discrepancy, g
                if metric >= best.get(child.state, (math.inf, math.inf)):
                    self.counters["pruned_dominance"] += 1
                    continue
                best[child.state] = metric
                if self.kind == "rank_discrepancy":
                    priority = discrepancy + 0.1 * g + 0.25 * child.h
                elif self.kind == "macro":
                    priority = (
                        0.2 * g + max(self.weight, 2.5) * child.h + 1.5 * child.bias
                    )
                    priority -= 2.0 * child.policy
                else:
                    priority = (
                        g + self.weight * child.h + child.bias - 2.0 * child.policy
                    )
                heapq.heappush(
                    queue,
                    (
                        priority,
                        discrepancy,
                        g,
                        next(self.counter),
                        child.state,
                        Link(link, child.pushes),
                    ),
                )
                self.frontier_size = len(queue)
                self.frontier_peak = max(self.frontier_peak, len(queue))
            self.peak = max(self.peak, len(best))
            ctx.peak_frontier = max(ctx.peak_frontier, len(queue))

    def beam_search(self):
        if self.context.beam_mode == "serial":
            yield from self.serial_beam_search()
            return
        ctx = self.context
        plans = yield from ctx.beam_plans(ctx.start)
        # A focused lane retains the established beam policy. A separate,
        # narrower lane explores diverse commitments without resetting either
        # frontier. Both lanes debit this session and the same global budget.
        anchor = self.serial_beam_search()
        anchor_done = False
        pending = iter(enumerate(plans))
        self.goals = []

        def activate(count):
            for _ in range(count):
                try:
                    index, plan = next(pending)
                except StopIteration:
                    return
                self.goals.append(GoalSession(self, plan, index, narrow=True))

        activate(8)
        cursor = 0
        exploration_turns = 0
        while not anchor_done or any(not g.done for g in self.goals):
            if not anchor_done:
                for _ in range(3):
                    spent = 0.0
                    before = self.nodes
                    try:
                        while True:
                            tick = time.perf_counter()
                            item = next(anchor)
                            spent += time.perf_counter() - tick
                            if isinstance(item, Found):
                                yield item
                                return
                            yield None
                            if self.nodes - before >= 32 or spent >= 0.025:
                                break
                    except StopIteration:
                        anchor_done = True
                        break
            if not self.goals or all(g.done for g in self.goals):
                before = len(self.goals)
                activate(8)
                if len(self.goals) == before:
                    if anchor_done:
                        return
                    continue
            for _ in range(len(self.goals)):
                goal = self.goals[cursor % len(self.goals)]
                cursor += 1
                if not goal.done:
                    break
            quantum = 0.025 if anchor_done else 0.0125
            if self.slice_deadline is not None:
                quantum = min(
                    quantum, max(0.0, self.slice_deadline - time.perf_counter())
                )
            expansions = (
                8
                if self.slice_node_limit is None
                else min(8, max(1, self.slice_node_limit - self.nodes))
            )
            found = goal.advance(quantum=quantum, expansions=expansions)
            if found is not None:
                yield found
                return
            exploration_turns += 1
            if all(g.done or g.nodes >= 64 for g in self.goals):
                activate(4)
            yield from ctx.checkpoint()

    def serial_beam_search(self):
        """Reference allocation uses the same per-goal transition kernel."""
        ctx = self.context
        if ctx.beam_mode == "interleaved":
            plans = ctx.initial_anchor_plans
        else:
            _, plans = yield from ctx.score(ctx.start)
        for phase, (width, depth_limit) in enumerate(
            ((48, 60), (96, 90), (160, 120), (256, 160))
        ):
            for index, plan in enumerate(plans):
                goal = GoalSession(self, plan, index, widths=((width, depth_limit),))
                goal.restarts = phase
                self.anchor_goals.append(goal)
                while True:
                    tick = time.perf_counter()
                    try:
                        item = next(goal.iterator)
                    except StopIteration:
                        goal.done = True
                        break
                    goal.elapsed += time.perf_counter() - tick
                    self.frontier_size = goal.frontier_size
                    self.frontier_peak = max(self.frontier_peak, goal.frontier_peak)
                    yield item
                    if isinstance(item, Found):
                        return

    def advance(self, *, quantum=0.25, expansions=128):
        started = time.perf_counter()
        cpu = time.thread_time()
        counters = self.context.metrics.copy()
        before = self.nodes
        self.slice_node_limit = before + expansions
        self.slice_deadline = started + quantum
        try:
            while not self.done:
                item = next(self.iterator)
                if isinstance(item, Found):
                    self.found = item
                    self.done = True
                    return item
                if (
                    self.nodes - before >= expansions
                    or time.perf_counter() - started >= quantum
                ):
                    return None
        except StopIteration:
            self.done = True
        finally:
            self.elapsed += time.perf_counter() - started
            self.cpu += time.thread_time() - cpu
            self.counters.update(
                {
                    k: v - counters[k]
                    for k, v in self.context.metrics.items()
                    if v != counters[k]
                }
            )
        return None


class GoalSession:
    """A commitment owns its frontier, depth, novelty and dominance tables."""

    def __init__(self, parent, plan, index, *, narrow=False, widths=None):
        self.parent, self.plan, self.index = parent, plan, index
        self.widths = widths or (
            ((8, 60), (24, 90), (48, 120), (96, 160))
            if narrow
            else ((48, 60), (96, 90), (160, 120), (256, 160))
        )
        self.nodes = 0
        self.peak = 1
        self.frontier_size = 1
        self.frontier_peak = 1
        self.elapsed = 0.0
        self.done = False
        self.depth = 0
        self.restarts = 0
        self.iterator = self.search()

    def search(self):
        ctx = self.parent.context
        plan = self.plan
        for phase, (width, depth_limit) in enumerate(self.widths):
            self.restarts += int(phase > 0)
            seen = {ctx.start: 0}
            frontier = [(0.0, 0, next(self.parent.counter), ctx.start, None)]
            important = plan.line | plan.route | plan.stands | (1 << plan.target)
            for _depth in range(depth_limit):
                if not frontier:
                    break
                candidates = []
                signatures = Counter()
                for _, cost, _, current, link in frontier:
                    yield from ctx.checkpoint()
                    self.parent.expand()
                    self.nodes += 1
                    won, target = ctx.goal(current)
                    if won:
                        yield Found(current, target, link)
                        return
                    children = yield from ctx.children(current, True)
                    parent_h, _ = yield from ctx.score(current, limit=1)
                    scored = []
                    for child in children:
                        yield from ctx.checkpoint()
                        g = cost + len(child.pushes)
                        if g >= seen.get(child.state, math.inf):
                            continue
                        bias = ctx.commitment_bias(plan, current, child, parent_h)
                        score = (
                            child.h
                            + 1.2 * (child.bias + 1.5 * bias)
                            - 2.0 * child.policy
                            + 0.08 * len(child.pushes)
                        )
                        push = child.pushes[0]
                        dest = ctx.push_lookup[push.cell, push.move][1]
                        relevant = (
                            bias < 0
                            or important & ((1 << push.cell) | (1 << dest))
                            or child.h <= parent_h
                        )
                        signature = (
                            child.state[0],
                            child.state[1] & important,
                            bool(child.state[2] & (1 << plan.target)),
                            int(child.h // 2) if math.isfinite(child.h) else -1,
                        )
                        scored.append(
                            (
                                relevant,
                                (
                                    score,
                                    g,
                                    next(self.parent.counter),
                                    child.state,
                                    Link(link, child.pushes),
                                    signature,
                                ),
                            )
                        )
                    # Commitment filtering belongs only to this session.
                    # The three unfiltered searches retain every primitive.
                    relevant = [row for keep, row in scored if keep]
                    escape = sorted(
                        (row for keep, row in scored if not keep),
                        key=lambda row: (row[0], row[2]),
                    )[:8]
                    candidates.extend(
                        relevant + escape if relevant else [row for _, row in scored]
                    )
                candidates.sort(key=lambda c: (c[0], c[2]))
                diversified = []
                for score, g, tie, state, link, signature in candidates:
                    novelty = signatures[signature]
                    signatures[signature] += 1
                    diversified.append(
                        (
                            score + 0.25 * novelty + 0.75 * max(0, novelty - 1),
                            g,
                            tie,
                            state,
                            link,
                            signature,
                        )
                    )
                diversified.sort(key=lambda c: (c[0], c[2]))
                frontier = []
                for score, g, tie, state, link, signature in diversified:
                    yield from ctx.checkpoint()
                    if g >= seen.get(state, math.inf):
                        continue
                    seen[state] = g
                    won, target = ctx.goal(state)
                    if won:
                        yield Found(state, target, link)
                        return
                    frontier.append((score, g, tie, state, link))
                    if len(frontier) >= width:
                        break
                self.peak = max(self.peak, len(seen))
                self.parent.peak = max(self.parent.peak, self.peak)
                self.frontier_size = len(frontier)
                self.frontier_peak = max(self.frontier_peak, len(frontier))
                ctx.peak_frontier = max(ctx.peak_frontier, len(frontier))

    def advance(self, *, quantum=0.0125, expansions=8):
        started = time.perf_counter()
        before = self.nodes
        try:
            while not self.done:
                item = next(self.iterator)
                if isinstance(item, Found):
                    self.done = True
                    return item
                if (
                    self.nodes - before >= expansions
                    or time.perf_counter() - started >= quantum
                ):
                    return None
        except StopIteration:
            self.done = True
        finally:
            self.elapsed += time.perf_counter() - started
        return None


def improve_keystrokes(ctx, incumbent, *, weight=4.0):
    """Anytime exact-player-position search with V3's keystroke cost objective."""
    counter = itertools.count()
    h, _ = yield from ctx.score(ctx.start, limit=1)
    queue = [(weight * h, 0, next(counter), ctx.start, ctx.player, None)]
    best = {(ctx.start, ctx.player): 0}
    best_length = len(incumbent[0])
    last_improvement = None
    while queue:
        yield from ctx.checkpoint()
        if (
            last_improvement is not None
            and time.perf_counter() - last_improvement >= 1.0
        ):
            return
        _, cost, _, state, player, link = heapq.heappop(queue)
        if cost != best.get((state, player)) or cost >= best_length:
            continue
        if ctx.max_nodes is not None and ctx.expanded >= ctx.max_nodes:
            raise BudgetExpired("node_cap")
        ctx.expanded += 1
        won, target = ctx.goal(state)
        if won:
            found = Found(state, target, link)
            moves, pushes = ctx.reconstruct(found, ctx.deadline)
            if len(moves) < best_length:
                best_length = len(moves)
                last_improvement = time.perf_counter()
                yield moves, pushes
        children = yield from ctx.children(state, True)
        for child in children:
            yield from ctx.checkpoint()
            os, xs = state[:2]
            actual, segment = player, 0
            for push in child.pushes:
                stand, dest = ctx.push_lookup[push.cell, push.move]
                segment += len(ctx.walk(actual, stand, os | xs)) + 1
                if push.piece == "O":
                    os = (os ^ (1 << push.cell)) | (1 << dest)
                else:
                    xs = (xs ^ (1 << push.cell)) | (1 << dest)
                actual = push.cell
            g = cost + segment
            key = child.state, actual
            if g >= best_length or g >= best.get(key, math.inf):
                continue
            best[key] = g
            priority = g + weight * child.h + 1.5 * child.bias
            heapq.heappush(
                queue,
                (
                    priority,
                    g,
                    next(counter),
                    child.state,
                    actual,
                    Link(link, child.pushes),
                ),
            )


def runtime_environment():
    limits = {}
    for name, path in [
        ("cpu_max", "/sys/fs/cgroup/cpu.max"),
        ("memory_max", "/sys/fs/cgroup/memory.max"),
    ]:
        try:
            limits[name] = Path(path).read_text().strip()
        except OSError:
            pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
        "cgroup": limits,
    }


def solve_v4(
    board,
    *,
    weight=2.0,
    max_nodes=500000,
    timeout_seconds=30.0,
    quality_timeout_seconds=10.0,
    schedule="rank",
    route_limit=3,
    strategy=None,
    ranker_path=None,
    beam_mode="interleaved",
    macro_mode="straight",
) -> PushSolveResult:
    started, cpu = time.perf_counter(), time.thread_time()
    if beam_mode not in ("serial", "interleaved"):
        raise ValueError("Unknown beam mode")
    if macro_mode not in ("straight", "forced"):
        raise ValueError("Unknown macro mode")
    if schedule not in SCHEDULES:
        raise ValueError("Unknown V4 schedule")
    if strategy is not None and strategy not in STRATEGIES:
        raise ValueError("Unknown V4 strategy")
    if route_limit not in (1, 2, 3):
        raise ValueError("V4 route_limit must be 1, 2, or 3")
    deadline = None if timeout_seconds is None else started + max(0, timeout_seconds)
    reserve = (
        0 if timeout_seconds is None else min(0.25, max(0, timeout_seconds) * 0.05)
    )
    search_deadline = None if deadline is None else deadline - reserve
    context = Context(board, search_deadline, max_nodes, route_limit=route_limit)
    context.beam_mode = beam_mode
    context.macro_mode = macro_mode
    sessions = []
    incumbent = None
    verification_ms = 0.0
    reason = "exhausted"
    winner = None
    try:
        context.check()
        if context.x_loss(context.xs):
            reason = "x_loss"
        else:
            for _ in context.initialize():
                pass
            if ranker_path is not None:
                from solver.push_solver.v4_ranker import FeatureRanker

                context.ranker = FeatureRanker(ranker_path, context.board)
                context.check()
                context.preprocess_ms = (time.perf_counter() - started) * 1000
            if (yieldless := context.goal(context.start))[0]:
                incumbent = Found(context.start, yieldless[1], None)
                winner = "precheck"
            else:
                sessions = [
                    Session(context, kind, weight=weight)
                    for kind in (STRATEGIES if strategy is None else (strategy,))
                ]
                cycle = [
                    session
                    for session, n in zip(
                        sessions, SCHEDULES[schedule] if strategy is None else (1,)
                    )
                    for _ in range(n)
                ]
                while any(not s.done for s in sessions):
                    for session in cycle:
                        if session.done:
                            continue
                        found = session.advance()
                        if found is not None:
                            t = time.perf_counter()
                            moves, pushes = context.reconstruct(found, deadline)
                            verified = verify_solution(context.board, moves)
                            verification_ms += (time.perf_counter() - t) * 1000
                            if verified.ok and (
                                deadline is None or time.perf_counter() <= deadline
                            ):
                                incumbent = (moves, pushes, verified.final_board)
                                winner = session.kind
                                break
                            context.metrics["invalid_candidates"] += 1
                    if incumbent is not None:
                        break
    except BudgetExpired as exc:
        reason = str(exc)
    if isinstance(incumbent, Found):
        t = time.perf_counter()
        moves, pushes = context.reconstruct(incumbent, deadline)
        checked = verify_solution(context.board, moves)
        verification_ms += (time.perf_counter() - t) * 1000
        incumbent = (
            (moves, pushes, checked.final_board)
            if checked.ok and (deadline is None or time.perf_counter() <= deadline)
            else None
        )
    baseline = len(incumbent[0]) if incumbent is not None else None
    quality_nodes = 0
    improved = False
    # The V3 keystroke objective, using cooperative V4 successors and the same
    # shared node/deadline counters. The verified incumbent survives interruption.
    quality_started = time.perf_counter()
    quality_end = (
        None
        if quality_timeout_seconds is None
        else started + max(0, quality_timeout_seconds)
    )
    if search_deadline is not None:
        quality_end = (
            min(quality_end, search_deadline)
            if quality_end is not None
            else search_deadline
        )
    search_nodes = context.expanded
    search_verification_ms = verification_ms
    if incumbent is not None and incumbent[0] and quality_timeout_seconds != 0:
        context.deadline = quality_end
        try:
            optimizer = improve_keystrokes(context, incumbent, weight=max(4.0, weight))
            while True:
                candidate = next(optimizer)
                if candidate is None:
                    continue
                t = time.perf_counter()
                checked = verify_solution(context.board, candidate[0])
                verification_ms += (time.perf_counter() - t) * 1000
                if checked.ok and len(candidate[0]) < len(incumbent[0]):
                    incumbent = candidate[0], candidate[1], checked.final_board
                    improved = True
        except (StopIteration, BudgetExpired):
            pass
        except Exception:
            context.metrics["optimizer_errors"] += 1
    quality_nodes = context.expanded - search_nodes
    quality_ms = (time.perf_counter() - quality_started) * 1000 - (
        verification_ms - search_verification_ms
    )
    elapsed = (time.perf_counter() - started) * 1000
    attempts = tuple(
        SearchAttempt(
            s.kind,
            s.kind == winner,
            s.nodes,
            s.peak,
            s.elapsed * 1000,
            None
            if s.kind == winner
            else (
                "incumbent_found"
                if incumbent and not s.done
                else reason
                if not s.done
                else "exhausted"
            ),
        )
        for s in sessions
    )
    diagnostics = {
        "strategies": {
            s.kind: {
                "expansions": s.nodes,
                "wall_ms": s.elapsed * 1000,
                "cpu_ms": s.cpu * 1000,
                "peak_closed": s.peak,
                "frontier_size": s.frontier_size,
                "peak_frontier": s.frontier_peak,
                "counters": dict(s.counters),
            }
            for s in sessions
        },
        "model_hashes": {"candidate": context.ranker.hash} if context.ranker else {},
        "cpu_clock": "thread_time",
        "version": "push-v4",
        "schedule": schedule,
        "route_limit": route_limit,
        "beam_mode": beam_mode,
        "macro_mode": macro_mode,
        "commitments": [
            {
                "index": g.index,
                "line": g.plan.line,
                "target": g.plan.target,
                "expansions": g.nodes,
                "wall_ms": g.elapsed * 1000,
                "peak_closed": g.peak,
                "done": g.done,
                "restarts": g.restarts,
            }
            for session in sessions
            for g in session.goals + session.anchor_goals
        ],
        "exact_hints": False,
        "preprocessing_ms": context.preprocess_ms,
        "verification_ms": verification_ms,
        "search_ms": max(
            0, elapsed - context.preprocess_ms - verification_ms - quality_ms
        ),
        "wall_ms": elapsed,
        "cpu_ms": (time.thread_time() - cpu) * 1000,
        "expanded_nodes": search_nodes,
        "quality_nodes": quality_nodes,
        "quality_ms": quality_ms,
        "peak_frontier": context.peak_frontier,
        "counters": dict(context.metrics),
        "runtime": runtime_environment(),
        "limits": {"seconds": timeout_seconds, "nodes": max_nodes},
        "cache_sizes": {
            "reverse": len(context.reverse_cache),
            "geometry": len(context.geometry_cache),
            "score": len(context.score_cache),
            "deadlock": len(context.deadlock_cache),
            "edges": len(context.edge_cache),
            "successors": len(context.successor_cache),
        },
    }
    result = PushSolveResult(
        solved=incumbent is not None,
        moves=incumbent[0] if incumbent else None,
        final_board=incumbent[2] if incumbent else None,
        pushes=incumbent[1] if incumbent else (),
        nodes_expanded=context.expanded,
        peak_closed_size=max((s.peak for s in sessions), default=1),
        elapsed_ms=elapsed,
        failure_reason=None if incumbent else reason,
        strategy=winner,
        attempts=attempts,
        baseline_keystrokes=baseline,
        quality_improved=improved,
        quality_nodes_expanded=quality_nodes,
        diagnostics=diagnostics,
    )
    # Break suspended-generator references promptly in warm worker processes.
    context.successor_cache.clear()
    return result
