"""Legal one-push successor generation."""

from __future__ import annotations

import math
from collections.abc import Callable

from solver.push_solver.deadlocks import is_deadlock, is_x_loss
from solver.push_solver.heuristics import (
    _legal_o_push_count_for,
    _plan_specific_push_bias,
    _priority_bias,
    _top_line_plans_for,
    goal_info,
)
from solver.push_solver.models import LinePlan, Push, State, StaticBoard
from solver.push_solver.state import _normalize_with_region, reachable


NormalizeWithRegion = Callable[
    [int, frozenset[int], frozenset[int], StaticBoard],
    tuple[State, frozenset[int]],
]
ReachableFn = Callable[[int, frozenset[int], frozenset[int], StaticBoard], frozenset[int]]


def successors(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int] | None = None,
    parent_h: float | None = None,
    target_access_penalty: float = 0.0,
    plan_cache: dict[State, LinePlan | None] | None = None,
    top_plan_cache: dict[State, tuple[LinePlan, ...]] | None = None,
    o_push_count_cache: dict[State, int] | None = None,
    candidate_cache: dict[tuple[frozenset[int], frozenset[int]], tuple] | None = None,
    push_reach_cache: dict[tuple[int, int, frozenset[int]], bool] | None = None,
    floor_reach_cache: dict[tuple[int, int, frozenset[int]], bool] | None = None,
    normalize_with_region: NormalizeWithRegion = _normalize_with_region,
    reachable_fn: ReachableFn = reachable,
) -> list[tuple[Push, State, frozenset[int], float, float]]:
    if region is None:
        region = reachable_fn(state.player, state.os, state.xs, board)
    occupied = state.os | state.xs
    parent_top_plans = _top_line_plans_for(
        state,
        board,
        region=region,
        target_access_penalty=target_access_penalty,
        top_plan_cache=top_plan_cache,
        candidate_cache=candidate_cache,
    )
    parent_plan = parent_top_plans[0] if parent_top_plans else None
    if plan_cache is not None and state not in plan_cache:
        plan_cache[state] = parent_plan
    if parent_h is None:
        parent_h = math.inf if parent_plan is None else parent_plan.score
    parent_o_push_count = _legal_o_push_count_for(
        state,
        board,
        region,
        o_push_count_cache,
    )

    results: list[tuple[Push, State, frozenset[int], float, float]] = []

    for piece, cells in (("O", state.os), ("X", state.xs)):
        for cell in sorted(cells):
            for move, stand, dest in board.push_transitions[cell]:
                if stand not in region:
                    continue
                if dest in occupied:
                    continue

                if piece == "O":
                    new_os = frozenset((state.os - {cell}) | {dest})
                    new_xs = state.xs
                else:
                    new_os = state.os
                    new_xs = frozenset((state.xs - {cell}) | {dest})
                    if is_x_loss(new_xs, board):
                        continue

                if is_deadlock(
                    new_os,
                    new_xs,
                    board,
                    player=cell,
                    push_reach_cache=push_reach_cache,
                    floor_reach_cache=floor_reach_cache,
                ):
                    continue

                next_state, next_region = normalize_with_region(cell, new_os, new_xs, board)

                child_o_push_count = _legal_o_push_count_for(
                    next_state,
                    board,
                    next_region,
                    o_push_count_cache,
                )

                if goal_info(next_state, board, region=next_region) is not None:
                    child_plan = None
                    h = 0.0
                else:
                    child_top_plans = _top_line_plans_for(
                        next_state,
                        board,
                        region=next_region,
                        target_access_penalty=target_access_penalty,
                        top_plan_cache=top_plan_cache,
                        candidate_cache=candidate_cache,
                    )
                    child_plan = child_top_plans[0] if child_top_plans else None
                    if plan_cache is not None and next_state not in plan_cache:
                        plan_cache[next_state] = child_plan
                    h = math.inf if child_plan is None else child_plan.score
                push = Push(piece=piece, cell=cell, move=move)

                bias = _priority_bias(
                    parent_state=state,
                    parent_region=region,
                    parent_h=parent_h,
                    parent_plan=parent_plan,
                    parent_o_push_count=parent_o_push_count,
                    child_o_push_count=child_o_push_count,
                    push=push,
                    child_state=next_state,
                    child_region=next_region,
                    child_h=h,
                    child_plan=child_plan,
                    board=board,
                )
                if parent_top_plans:
                    bias += min(
                        _plan_specific_push_bias(
                            plan=plan,
                            push=push,
                            parent_region=region,
                            child_region=next_region,
                            board=board,
                        )
                        for plan in parent_top_plans
                    )
                results.append((push, next_state, next_region, h, bias))

    results.sort(key=lambda item: (item[3] + item[4], item[3]))
    return results

