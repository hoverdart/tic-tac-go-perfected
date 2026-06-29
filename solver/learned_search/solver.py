"""Experimental solver using learned child-path ranking."""

from __future__ import annotations

import heapq
import itertools
import os

from solver import optimized_solver
from solver.board_utils import normalize_board
from solver.learned_search.features import candidate_features
from solver.learned_search.linear_ranker import LinearRanker


def solve(
    start_board,
    *,
    ranker: LinearRanker | None = None,
    learned_weight: float = 1.0,
    progress_every: int = 100_000,
    max_states: int | None = None,
    mode: str | None = None,
):
    """Run weighted A* with an extra learned child-ranking term.

    This is an experimental path. The model only changes candidate ordering; all
    legality, pruning, parent tracking, and final replay validation still come
    from the optimized solver.
    """
    ranker = ranker or LinearRanker.default()
    start_board = normalize_board(start_board)
    geometry = optimized_solver._geometry_for_board(start_board)
    start_key = optimized_solver._to_key(start_board)
    mode = (mode or os.getenv("SOLVER_MODE") or "hybrid").strip().lower()
    if mode not in {"hybrid", "fast", "exact"}:
        mode = "hybrid"

    if optimized_solver._is_solved(start_key, geometry):
        return "", start_board, 1
    if optimized_solver._pruned(start_key, geometry):
        return None, None, 1

    queue = []
    counter = itertools.count()
    parents = {start_key: optimized_solver.Parent(previous=None, segment="")}
    best_cost_seen = {start_key: 0}
    states_checked = 0
    best_solution = None
    weight = optimized_solver._weight_for_mode(mode)

    heapq.heappush(
        queue,
        (optimized_solver._heuristic(start_key, geometry) * weight, 0, next(counter), start_key),
    )

    while queue:
        _priority, cost_so_far, _, current_key = heapq.heappop(queue)
        if cost_so_far != best_cost_seen.get(current_key):
            continue
        if best_solution and cost_so_far >= len(best_solution[0]):
            continue

        states_checked += 1
        if progress_every and states_checked % progress_every == 0:
            print(states_checked, flush=True)

        if optimized_solver._pruned(current_key, geometry):
            if max_states is not None and states_checked >= max_states:
                break
            continue

        if optimized_solver._is_solved(current_key, geometry):
            moves = optimized_solver._reconstruct(parents, current_key)
            validated = optimized_solver._validated_result(start_board, geometry, moves)
            if validated is not None:
                solution_moves, final_board = validated
                best_solution = (solution_moves, final_board, states_checked)
                if mode == "fast":
                    break
                if mode == "hybrid":
                    weight = 1.0
            if max_states is not None and states_checked >= max_states:
                break
            continue

        if max_states is not None and states_checked >= max_states:
            break

        for next_key, segment in optimized_solver._next_states(current_key, geometry):
            if optimized_solver._pruned(next_key, geometry):
                continue

            next_cost = cost_so_far + len(segment)
            if best_solution and next_cost >= len(best_solution[0]):
                continue
            if next_cost >= best_cost_seen.get(next_key, 1_000_000_000):
                continue

            features = candidate_features(current_key, next_key, segment, geometry)
            learned_score = ranker.score(features)
            best_cost_seen[next_key] = next_cost
            parents[next_key] = optimized_solver.Parent(previous=current_key, segment=segment)
            next_priority = (
                next_cost
                + (optimized_solver._heuristic(next_key, geometry) * weight)
                - (learned_weight * learned_score)
            )
            heapq.heappush(queue, (next_priority, next_cost, next(counter), next_key))

    if best_solution is None:
        return None, None, states_checked
    return best_solution
