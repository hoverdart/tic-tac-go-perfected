"""V3 anytime keystroke optimizer layered on top of a verified V2 solution.

V2 intentionally merges every player position in the same reachable region and
therefore cannot price the walking needed between pushes.  V3 keeps that
compressed box state for successor generation, but adds the player's exact
post-push cell to the quality-search key.  This makes the path cost equal to the
number of direction keystrokes while retaining push-level branching.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass

from solver.push_solver.models import (
    DIRECTION_BY_MOVE,
    GoalInfo,
    Push,
    State,
    StaticBoard,
)
from solver.push_solver.reconstruction import reconstruct_moves
from solver.push_solver.search import _build_search_context, _strategy_children_for


QualityKey = tuple[State, int]


@dataclass(frozen=True)
class QualityParent:
    previous: QualityKey
    pushes: tuple[Push, ...]


@dataclass(frozen=True)
class QualitySearchResult:
    improved: bool
    moves: str
    final_board: tuple[tuple[str, ...], ...]
    pushes: tuple[Push, ...]
    nodes_expanded: int
    peak_closed_size: int
    elapsed_ms: float
    termination_reason: str


def _walk_distances(
    player: int,
    os: frozenset[int],
    xs: frozenset[int],
    static: StaticBoard,
) -> dict[int, int]:
    """Return shortest walking distances without pushing any piece."""
    blocked = os | xs
    queue = [player]
    distances = {player: 0}
    for current in queue:
        next_distance = distances[current] + 1
        for nxt in static.adjacency[current]:
            if nxt in blocked or nxt in distances:
                continue
            distances[nxt] = next_distance
            queue.append(nxt)
    return distances


def _push_stand_and_destination(
    push: Push,
    static: StaticBoard,
) -> tuple[int, int]:
    dr, dc = DIRECTION_BY_MOVE[push.move]
    row, col = static.coord(push.cell)
    return (
        static.index(row - dr, col - dc),
        static.index(row + dr, col + dc),
    )


def _segment_cost(
    state: State,
    actual_player: int,
    pushes: tuple[Push, ...],
    static: StaticBoard,
    initial_distances: dict[int, int],
) -> tuple[int, int] | None:
    """Price a one-push or forced-macro edge in actual keystrokes."""
    os = state.os
    xs = state.xs
    cost = 0
    distances = initial_distances

    for index, push in enumerate(pushes):
        stand, destination = _push_stand_and_destination(push, static)
        walk = distances.get(stand)
        if walk is None:
            return None
        cost += walk + 1

        if push.piece == "O":
            os = frozenset((os - {push.cell}) | {destination})
        else:
            xs = frozenset((xs - {push.cell}) | {destination})
        actual_player = push.cell

        if index + 1 < len(pushes):
            distances = _walk_distances(actual_player, os, xs, static)

    return cost, actual_player


def _goal_options(
    state: State,
    distances: dict[int, int],
    static: StaticBoard,
) -> tuple[tuple[int, GoalInfo], ...]:
    options: list[tuple[int, GoalInfo]] = []
    for line in static.win_lines:
        os_in_line = [cell for cell in line if cell in state.os]
        if len(os_in_line) != 2:
            continue
        player_target = next(cell for cell in line if cell not in state.os)
        if player_target in state.xs or player_target not in distances:
            continue
        options.append(
            (
                distances[player_target],
                GoalInfo(line=line, player_target=player_target),
            )
        )
    return tuple(options)


def _reconstruct_quality_pushes(
    parents: dict[QualityKey, QualityParent],
    goal: QualityKey,
) -> tuple[Push, ...]:
    chunks: list[tuple[Push, ...]] = []
    current = goal
    while current in parents:
        parent = parents[current]
        chunks.append(parent.pushes)
        current = parent.previous
    return tuple(push for chunk in reversed(chunks) for push in chunk)


def improve_solution(
    *,
    static: StaticBoard,
    start_state: State,
    initial_player: int,
    incumbent_moves: str,
    incumbent_final_board: tuple[tuple[str, ...], ...],
    incumbent_pushes: tuple[Push, ...],
    max_nodes: int | None,
    deadline: float | None,
    weight: float = 4.0,
    bias_scale: float = 1.5,
    stagnation_seconds: float = 1.0,
) -> QualitySearchResult:
    """Search for a shorter verified-path candidate without risking coverage.

    The caller already owns a verified V2 incumbent.  This function can only
    return that path or a strictly shorter reconstructed path; failure and
    timeout therefore leave board coverage unchanged.
    """
    started = time.perf_counter()
    context = _build_search_context(static, start_state, initial_player)
    start_key: QualityKey = (start_state, initial_player)
    counter = itertools.count()
    queue: list[tuple[float, int, int, QualityKey]] = []
    best_cost_by_key: dict[QualityKey, int] = {start_key: 0}
    parents: dict[QualityKey, QualityParent] = {}
    heapq.heappush(
        queue,
        (weight * context.h_cache[start_state], 0, next(counter), start_key),
    )

    initial_length = len(incumbent_moves)
    best_length = initial_length
    best_moves = incumbent_moves
    best_final_board = incumbent_final_board
    best_pushes = incumbent_pushes
    nodes_expanded = 0
    peak_closed_size = 1
    termination_reason = "exhausted"
    last_improvement_at: float | None = None

    while queue:
        now = time.perf_counter()
        if deadline is not None and now >= deadline:
            termination_reason = "timeout"
            break
        if (
            last_improvement_at is not None
            and now - last_improvement_at >= stagnation_seconds
        ):
            termination_reason = "quality_stable"
            break
        if max_nodes is not None and nodes_expanded >= max_nodes:
            termination_reason = "node_cap"
            break

        _priority, cost, _tie, key = heapq.heappop(queue)
        if cost != best_cost_by_key.get(key) or cost >= best_length:
            continue

        state, actual_player = key
        nodes_expanded += 1
        distances = _walk_distances(
            actual_player,
            state.os,
            state.xs,
            static,
        )

        options = _goal_options(state, distances, static)
        if options:
            final_walk_cost, goal = min(options, key=lambda item: item[0])
            if cost + final_walk_cost < best_length:
                pushes = _reconstruct_quality_pushes(parents, key)
                moves, final_board = reconstruct_moves(
                    pushes,
                    goal,
                    static,
                    start_state,
                    initial_player,
                )
                if len(moves) < best_length:
                    best_length = len(moves)
                    best_moves = moves
                    best_final_board = final_board
                    best_pushes = pushes
                    last_improvement_at = time.perf_counter()

        children = _strategy_children_for(
            context,
            state,
            use_macros=True,
            bias_scale=bias_scale,
            policy_weight=0.0,
        )
        for pushes, nxt, child_region, child_h, bias, _push_cost, _policy in children:
            priced = _segment_cost(
                state,
                actual_player,
                pushes,
                static,
                distances,
            )
            if priced is None:
                continue
            segment_cost, next_actual_player = priced
            next_cost = cost + segment_cost
            if next_cost >= best_length:
                continue
            next_key = (nxt, next_actual_player)
            if next_cost >= best_cost_by_key.get(next_key, 1_000_000_000):
                continue
            context.region_cache.setdefault(nxt, child_region)
            context.h_cache.setdefault(nxt, child_h)
            best_cost_by_key[next_key] = next_cost
            parents[next_key] = QualityParent(previous=key, pushes=pushes)
            priority = (
                next_cost
                + (weight * child_h)
                + (bias_scale * bias)
            )
            heapq.heappush(
                queue,
                (priority, next_cost, next(counter), next_key),
            )
        peak_closed_size = max(peak_closed_size, len(best_cost_by_key))

    return QualitySearchResult(
        improved=best_length < initial_length,
        moves=best_moves,
        final_board=best_final_board,
        pushes=best_pushes,
        nodes_expanded=nodes_expanded,
        peak_closed_size=peak_closed_size,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        termination_reason=termination_reason,
    )
