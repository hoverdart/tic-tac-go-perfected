"""Line-plan heuristics and push-priority bias helpers."""

from __future__ import annotations

import math

from solver.push_solver.models import (
    DIRECTION_BY_MOVE,
    EMPTY_CELL_SET,
    EMPTY_ROUTE,
    GoalInfo,
    LinePlan,
    Push,
    State,
    StaticBoard,
)
from solver.push_solver.state import reachable


PENALTY_PER_BLOCKING_X: float = 1.0
PLAYER_TARGET_X_PENALTY: float = 2.0
PLAYER_TARGET_UNREACHABLE_PENALTY: float = 0.0
X_ON_LINE_PENALTY: float = 0.5
X_PUSH_BASE_BIAS: float = 0.35
O_PUSH_BASE_BIAS: float = -0.10
REACH_GAIN_BIAS: float = 1.80
TARGET_ACCESS_UNBLOCK_BIAS: float = 9.0
O_PUSH_ACCESS_GAIN_BIAS: float = 0.80
HEURISTIC_PROGRESS_BIAS: float = 1.00
PLAN_X_CLEAR_BIAS: float = 2.00
PLAN_TARGET_REACHABLE_BIAS: float = 2.50
PLAN_TARGET_UNREACHABLE_BIAS: float = 2.00


def goal_info(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int] | None = None,
) -> GoalInfo | None:
    region = region or reachable(state.player, state.os, state.xs, board)
    for line in board.win_lines:
        os_in_line = [cell for cell in line if cell in state.os]
        if len(os_in_line) != 2:
            continue
        player_target = next(cell for cell in line if cell not in state.os)
        if player_target in state.xs:
            continue
        if player_target in region:
            return GoalInfo(line=line, player_target=player_target)
    return None


def _route_cells(cell: int, target: int, board: StaticBoard) -> tuple[int, ...]:
    """Box-landing cells and player-stand cells along the canonical shortest
    wall-only push route from `cell` to `target` (both endpoints included)."""
    return board.push_routes.get(target, {}).get(cell, EMPTY_ROUTE)


def _stand_cells(cell: int, target: int, board: StaticBoard) -> tuple[int, ...]:
    return board.push_stand_routes.get(target, {}).get(cell, EMPTY_ROUTE)


def _occupancy_penalty(
    cell: int,
    target: int,
    blockers: frozenset[int],
    board: StaticBoard,
) -> float:
    base = board.push_distances.get(target, {}).get(cell, math.inf)
    if base == math.inf:
        return math.inf
    route_set = board.push_route_sets.get(target, {}).get(cell, EMPTY_CELL_SET)
    blocking = len(blockers & route_set)
    return base + (PENALTY_PER_BLOCKING_X * blocking)


def _top_line_plans(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int],
    target_access_penalty: float = 0.0,
    limit: int = 8,
) -> tuple[LinePlan, ...]:
    if len(state.os) < 2:
        return ()

    o_cells = tuple(state.os)
    xs = state.xs
    distances = board.push_distances
    route_sets = board.push_route_sets
    candidates: list[
        tuple[
            float,
            tuple[int, int, int],
            int,
            tuple[int, int],
            tuple[tuple[int, int], tuple[int, int]],
            int,
            int,
            int,
            int,
            int,
        ]
    ] = []
    for line in board.win_lines:
        x_on_line_count = sum(1 for cell in line if cell in xs)
        line_x_penalty = X_ON_LINE_PENALTY * x_on_line_count
        for player_target in line:
            targets = tuple(cell for cell in line if cell != player_target)
            if len(targets) != 2:
                continue

            plan_penalty = 0.0
            if player_target in xs:
                plan_penalty += PLAYER_TARGET_X_PENALTY
            if player_target not in region:
                plan_penalty += target_access_penalty
            plan_penalty += line_x_penalty

            assignments = (
                (o_cells[0], targets[0], o_cells[1], targets[1]),
                (o_cells[0], targets[1], o_cells[1], targets[0]),
            )
            for o_a, target_a, o_b, target_b in assignments:
                distance_a = distances.get(target_a, {}).get(o_a)
                distance_b = distances.get(target_b, {}).get(o_b)
                if distance_a is None or distance_b is None:
                    continue

                route_a = route_sets.get(target_a, {}).get(o_a, EMPTY_CELL_SET)
                route_b = route_sets.get(target_b, {}).get(o_b, EMPTY_CELL_SET)
                blocking_a = len(xs & route_a) + (1 if o_b in route_a else 0)
                blocking_b = len(xs & route_b) + (1 if o_a in route_b else 0)
                cost_a = distance_a + (PENALTY_PER_BLOCKING_X * blocking_a)
                cost_b = distance_b + (PENALTY_PER_BLOCKING_X * blocking_b)
                o_assignment = ((o_a, target_a), (o_b, target_b))
                score = cost_a + cost_b + plan_penalty

                candidates.append(
                    (
                        score,
                        line,
                        player_target,
                        targets,
                        o_assignment,
                        x_on_line_count,
                        o_a,
                        target_a,
                        o_b,
                        target_b,
                    )
                )

    candidates.sort(key=lambda item: item[0])
    plans: list[LinePlan] = []
    stand_route_sets = board.push_stand_route_sets
    for (
        score,
        line,
        player_target,
        targets,
        o_assignment,
        x_on_line_count,
        o_a,
        target_a,
        o_b,
        target_b,
    ) in candidates[:limit]:
        route_cells = (
            route_sets.get(target_a, {}).get(o_a, EMPTY_CELL_SET)
            | route_sets.get(target_b, {}).get(o_b, EMPTY_CELL_SET)
        )
        stand_cells = (
            stand_route_sets.get(target_a, {}).get(o_a, EMPTY_CELL_SET)
            | stand_route_sets.get(target_b, {}).get(o_b, EMPTY_CELL_SET)
        )
        blocked_route_count = len(state.xs & route_cells)
        plans.append(
            LinePlan(
                score=score,
                line=line,
                player_target=player_target,
                o_targets=targets,
                o_assignment=o_assignment,
                route_cells=route_cells,
                stand_cells=stand_cells,
                player_target_has_x=player_target in state.xs,
                player_target_reachable=player_target in region,
                x_on_line_count=x_on_line_count,
                blocked_route_count=blocked_route_count,
            )
        )

    return tuple(plans)


def _best_line_plan(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int] | None = None,
    target_access_penalty: float = 0.0,
) -> LinePlan | None:
    region = region or reachable(state.player, state.os, state.xs, board)
    plans = _top_line_plans(
        state,
        board,
        region=region,
        target_access_penalty=target_access_penalty,
        limit=1,
    )
    if not plans:
        return None
    return plans[0]


def heuristic(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int] | None = None,
    target_access_penalty: float = 0.0,
) -> float:
    if goal_info(state, board, region=region) is not None:
        return 0.0
    plan = _best_line_plan(
        state,
        board,
        region=region,
        target_access_penalty=target_access_penalty,
    )
    if plan is None:
        return math.inf
    return plan.score


def _line_plan_for(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int],
    target_access_penalty: float,
    plan_cache: dict[State, LinePlan | None] | None = None,
) -> LinePlan | None:
    if plan_cache is None:
        return _best_line_plan(
            state,
            board,
            region=region,
            target_access_penalty=target_access_penalty,
        )
    if state not in plan_cache:
        plan_cache[state] = _best_line_plan(
            state,
            board,
            region=region,
            target_access_penalty=target_access_penalty,
        )
    return plan_cache[state]


def _top_line_plans_for(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int],
    target_access_penalty: float,
    top_plan_cache: dict[State, tuple[LinePlan, ...]] | None = None,
    limit: int = 8,
) -> tuple[LinePlan, ...]:
    if top_plan_cache is None:
        return _top_line_plans(
            state,
            board,
            region=region,
            target_access_penalty=target_access_penalty,
            limit=limit,
        )
    if state not in top_plan_cache:
        top_plan_cache[state] = _top_line_plans(
            state,
            board,
            region=region,
            target_access_penalty=target_access_penalty,
            limit=limit,
        )
    return top_plan_cache[state]


def _push_distance(cell: int, target: int, board: StaticBoard) -> float:
    return board.push_distances.get(target, {}).get(cell, math.inf)


def _legal_o_push_count(
    state: State,
    board: StaticBoard,
    region: frozenset[int],
) -> int:
    occupied = state.os | state.xs
    count = 0
    for cell in state.os:
        for _move, stand, dest in board.push_transitions[cell]:
            if stand in region and dest not in occupied:
                count += 1
    return count


def _legal_o_push_count_for(
    state: State,
    board: StaticBoard,
    region: frozenset[int],
    o_push_count_cache: dict[State, int] | None = None,
) -> int:
    if o_push_count_cache is None:
        return _legal_o_push_count(state, board, region)
    if state not in o_push_count_cache:
        o_push_count_cache[state] = _legal_o_push_count(state, board, region)
    return o_push_count_cache[state]


def _push_destination(push: Push, board: StaticBoard) -> int:
    dr, dc = DIRECTION_BY_MOVE[push.move]
    row, col = board.coord(push.cell)
    return board.index(row + dr, col + dc)


def _plan_specific_push_bias(
    *,
    plan: LinePlan,
    push: Push,
    parent_region: frozenset[int],
    child_region: frozenset[int],
    board: StaticBoard,
) -> float:
    bias = 0.0
    dest = _push_destination(push, board)
    important_cells = set(plan.line) | set(plan.route_cells) | set(plan.stand_cells)

    if push.piece == "X":
        if push.cell == plan.player_target:
            bias -= PLAN_X_CLEAR_BIAS * 2.0
        elif push.cell in plan.line:
            bias -= PLAN_X_CLEAR_BIAS * 1.25
        elif push.cell in plan.route_cells:
            bias -= PLAN_X_CLEAR_BIAS
        elif push.cell in plan.stand_cells:
            bias -= PLAN_X_CLEAR_BIAS * 0.75

        if dest == plan.player_target:
            bias += PLAN_X_CLEAR_BIAS * 2.0
        elif dest in plan.line:
            bias += PLAN_X_CLEAR_BIAS * 1.25
        elif dest in plan.route_cells:
            bias += PLAN_X_CLEAR_BIAS
        elif dest in plan.stand_cells:
            bias += PLAN_X_CLEAR_BIAS * 0.75
    elif push.piece == "O":
        assigned_targets = {
            o_cell: target for o_cell, target in plan.o_assignment
        }
        target = assigned_targets.get(push.cell)
        if target is not None:
            before = _push_distance(push.cell, target, board)
            after = _push_distance(dest, target, board)
            if after < before:
                bias -= min(before - after, 3) * 0.5
            elif after > before:
                bias += min(after - before, 3) * 0.35
            if dest == target:
                bias -= PLAN_X_CLEAR_BIAS
        elif push.cell in important_cells:
            bias += 0.4

    if plan.player_target not in parent_region and plan.player_target in child_region:
        bias -= PLAN_TARGET_REACHABLE_BIAS
        bias -= TARGET_ACCESS_UNBLOCK_BIAS * 0.5
    elif plan.player_target not in child_region:
        bias += PLAN_TARGET_UNREACHABLE_BIAS * 0.25

    return bias


def _priority_bias(
    *,
    parent_state: State,
    parent_region: frozenset[int],
    parent_h: float,
    parent_plan: LinePlan | None,
    parent_o_push_count: int,
    child_o_push_count: int,
    push: Push,
    child_state: State,
    child_region: frozenset[int],
    child_h: float,
    child_plan: LinePlan | None,
    board: StaticBoard,
) -> float:
    del parent_state
    bias = X_PUSH_BASE_BIAS if push.piece == "X" else O_PUSH_BASE_BIAS
    if child_h < parent_h:
        bias -= HEURISTIC_PROGRESS_BIAS
    elif child_h > parent_h:
        bias += 0.25

    reach_gain = len(child_region) - len(parent_region)
    if reach_gain > 0:
        bias -= min(reach_gain, 10) * REACH_GAIN_BIAS

    o_push_gain = child_o_push_count - parent_o_push_count
    if o_push_gain > 0:
        bias -= min(o_push_gain, 4) * O_PUSH_ACCESS_GAIN_BIAS

    if push.piece == "X" and parent_plan is not None:
        dr, dc = DIRECTION_BY_MOVE[push.move]
        row, col = board.coord(push.cell)
        dest = board.index(row + dr, col + dc)
        important_cells = set(parent_plan.line) | set(parent_plan.route_cells)

        if push.cell == parent_plan.player_target:
            bias -= PLAN_X_CLEAR_BIAS
        elif push.cell in important_cells:
            bias -= PLAN_X_CLEAR_BIAS * 0.5

        if dest == parent_plan.player_target:
            bias += PLAN_X_CLEAR_BIAS
        elif dest in important_cells:
            bias += PLAN_X_CLEAR_BIAS * 0.5

    if parent_plan is not None and child_plan is not None:
        if not child_plan.player_target_reachable:
            bias += PLAN_TARGET_UNREACHABLE_BIAS
        if parent_plan.player_target_has_x and not child_plan.player_target_has_x:
            bias -= PLAN_X_CLEAR_BIAS
        if (
            not parent_plan.player_target_reachable
            and child_plan.player_target_reachable
        ):
            bias -= PLAN_TARGET_REACHABLE_BIAS
            bias -= TARGET_ACCESS_UNBLOCK_BIAS

    return bias

