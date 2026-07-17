"""Compatibility facade for the push-level Tic Tac Go solver.

The implementation is split into focused modules:

- ``models`` for shared data classes and constants,
- ``state`` for board parsing and state normalization,
- ``deadlocks`` for conservative dead-state checks,
- ``heuristics`` for line plans and priority bias,
- ``successors`` for legal one-push successor generation,
- ``reconstruction`` for concrete keystroke reconstruction,
- ``search`` for individual strategy execution,
- ``portfolio`` for production solve orchestration.

This module intentionally keeps the historical ``solver.push_solver.core`` API
stable for tests, diagnostics, and training tools.
"""

from __future__ import annotations

from solver.push_solver.deadlocks import (
    _assignment_survives_frozen_constraints,
    _cell_is_on_win_line,
    _direction_permanently_blocked,
    _floor_reachable_with_permanent_blockers,
    _has_viable_line_under_frozen_constraints,
    _piece_permanently_frozen,
    _push_reachable_with_permanent_blockers,
    frozen_pieces,
    is_deadlock,
    is_x_loss,
)
from solver.push_solver.heuristics import (
    HEURISTIC_PROGRESS_BIAS,
    O_PUSH_ACCESS_GAIN_BIAS,
    O_PUSH_BASE_BIAS,
    PENALTY_PER_BLOCKING_X,
    PLAN_TARGET_REACHABLE_BIAS,
    PLAN_TARGET_UNREACHABLE_BIAS,
    PLAN_X_CLEAR_BIAS,
    PLAYER_TARGET_UNREACHABLE_PENALTY,
    PLAYER_TARGET_X_PENALTY,
    REACH_GAIN_BIAS,
    TARGET_ACCESS_UNBLOCK_BIAS,
    X_ON_LINE_PENALTY,
    X_PUSH_BASE_BIAS,
    _best_line_plan,
    _legal_o_push_count,
    _legal_o_push_count_for,
    _line_plan_for,
    _occupancy_penalty,
    _plan_specific_push_bias,
    _priority_bias,
    _push_destination,
    _push_distance,
    _route_cells,
    _stand_cells,
    _top_line_plans,
    _top_line_plans_for,
    goal_info,
    heuristic,
)
from solver.push_solver.models import (
    DIRECTION_BY_MOVE,
    DIRECTIONS,
    EMPTY_CELL_SET,
    EMPTY_ROUTE,
    GoalInfo,
    LinePlan,
    Parent,
    Push,
    PushSolveResult,
    SearchAttempt,
    SearchContext,
    SearchStrategyConfig,
    State,
    StaticBoard,
    StrategyChild,
)
from solver.push_solver.portfolio import (
    _attempt_budget,
    _attempt_from_result,
    _portfolio_configs,
    _precheck_result,
    _result,
    solve,
    solve_v1,
    solve_v2,
)
from solver.push_solver.reconstruction import (
    _reconstruct_pushes,
    _shortest_walk,
    _state_to_board,
    reconstruct_moves,
)
from solver.push_solver.search import (
    _build_search_context,
    _macro_children_for,
    _policy_score_for,
    _push_items_for_piece_cell,
    _run_strategy,
    _same_piece_continuation,
    _strategy_children_for,
    _successors_for,
    _timed_out,
)
from solver.push_solver.state import parse_board, reachable
from solver.push_solver.successors import successors as _successors_impl


def _normalize_with_region(
    player: int,
    os: frozenset[int],
    xs: frozenset[int],
    board: StaticBoard,
) -> tuple[State, frozenset[int]]:
    region = reachable(player, os, xs, board)
    return State(player=min(region), os=os, xs=xs), region


def normalize_state(
    player: int,
    os: frozenset[int],
    xs: frozenset[int],
    board: StaticBoard,
) -> State:
    state, _region = _normalize_with_region(player, os, xs, board)
    return state


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
) -> list[tuple[Push, State, frozenset[int], float, float]]:
    return _successors_impl(
        state,
        board,
        region=region,
        parent_h=parent_h,
        target_access_penalty=target_access_penalty,
        plan_cache=plan_cache,
        top_plan_cache=top_plan_cache,
        o_push_count_cache=o_push_count_cache,
        normalize_with_region=_normalize_with_region,
        reachable_fn=reachable,
    )
