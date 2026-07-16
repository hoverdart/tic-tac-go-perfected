"""Search contexts, macro expansion, and individual push-search strategies."""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import defaultdict

from solver.push_solver.deadlocks import is_deadlock, is_x_loss
from solver.push_solver.heuristics import (
    _legal_o_push_count_for,
    _plan_specific_push_bias,
    _priority_bias,
    _push_destination,
    _push_distance,
    _top_line_plans_for,
    goal_info,
)
from solver.push_solver.models import (
    LinePlan,
    Parent,
    Push,
    PushSolveResult,
    SearchContext,
    SearchStrategyConfig,
    State,
    StaticBoard,
    StrategyChild,
)
from solver.push_solver.reconstruction import _reconstruct_pushes, reconstruct_moves
from solver.push_solver.state import _normalize_with_region, reachable
from solver.push_solver.successors import successors


def _result(
    *,
    solved: bool,
    moves: str | None,
    final_board: tuple[tuple[str, ...], ...] | None,
    pushes: tuple[Push, ...],
    nodes_expanded: int,
    peak_closed_size: int,
    started: float,
    failure_reason: str | None,
    strategy: str | None,
) -> PushSolveResult:
    return PushSolveResult(
        solved=solved,
        moves=moves,
        final_board=final_board,
        pushes=pushes,
        nodes_expanded=nodes_expanded,
        peak_closed_size=peak_closed_size,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        failure_reason=failure_reason,
        strategy=strategy,
    )


def _timed_out(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _build_search_context(
    static: StaticBoard,
    start_state: State,
    initial_player: int,
) -> SearchContext:
    start_region = reachable(start_state.player, start_state.os, start_state.xs, static)
    plan_cache: dict[State, LinePlan | None] = {}
    top_plan_cache: dict[State, tuple[LinePlan, ...]] = {}
    o_push_count_cache: dict[State, int] = {}
    start_o_push_count = _legal_o_push_count_for(
        start_state,
        static,
        start_region,
        o_push_count_cache,
    )
    target_access_penalty = (
        3.0
        if len(start_region) <= 2
        and start_o_push_count > 0
        else 0.0
    )
    start_top_plans = _top_line_plans_for(
        start_state,
        static,
        region=start_region,
        target_access_penalty=target_access_penalty,
        top_plan_cache=top_plan_cache,
    )
    start_plan = start_top_plans[0] if start_top_plans else None
    plan_cache[start_state] = start_plan
    start_h = math.inf if start_plan is None else start_plan.score
    return SearchContext(
        static=static,
        start_state=start_state,
        initial_player=initial_player,
        target_access_penalty=target_access_penalty,
        region_cache={start_state: start_region},
        h_cache={start_state: start_h},
        plan_cache=plan_cache,
        top_plan_cache=top_plan_cache,
        o_push_count_cache=o_push_count_cache,
        deadlock_cache={},
        successor_cache={},
        policy_score_cache={},
    )


def _is_deadlock_cached(
    context: SearchContext,
    os: frozenset[int],
    xs: frozenset[int],
    *,
    player: int | None,
) -> bool:
    key = (os, xs, player)
    if key not in context.deadlock_cache:
        context.deadlock_cache[key] = is_deadlock(os, xs, context.static, player=player)
    return context.deadlock_cache[key]


def _successors_for(
    context: SearchContext,
    state: State,
) -> tuple[tuple[Push, State, frozenset[int], float, float], ...]:
    if state not in context.successor_cache:
        region = context.region_cache[state]
        current_h = context.h_cache[state]
        items = tuple(
            successors(
                state,
                context.static,
                region=region,
                parent_h=current_h,
                target_access_penalty=context.target_access_penalty,
                plan_cache=context.plan_cache,
                top_plan_cache=context.top_plan_cache,
                o_push_count_cache=context.o_push_count_cache,
                deadlock_cache=context.deadlock_cache,
            )
        )
        context.successor_cache[state] = items
        for _push, nxt, child_region, child_h, _bias in items:
            context.region_cache.setdefault(nxt, child_region)
            context.h_cache.setdefault(nxt, child_h)
    return context.successor_cache[state]


def _push_items_for_piece_cell(
    context: SearchContext,
    state: State,
    *,
    piece: str,
    cell: int,
) -> list[tuple[Push, State, frozenset[int], float, float]]:
    static = context.static
    region = context.region_cache[state]
    parent_h = context.h_cache[state]
    occupied = state.os | state.xs
    parent_top_plans = _top_line_plans_for(
        state,
        static,
        region=region,
        target_access_penalty=context.target_access_penalty,
        top_plan_cache=context.top_plan_cache,
    )
    parent_plan = parent_top_plans[0] if parent_top_plans else None
    if state not in context.plan_cache:
        context.plan_cache[state] = parent_plan
    parent_o_push_count = _legal_o_push_count_for(
        state,
        static,
        region,
        context.o_push_count_cache,
    )

    items: list[tuple[Push, State, frozenset[int], float, float]] = []
    for move, stand, dest in static.push_transitions[cell]:
        if stand not in region or dest in occupied:
            continue

        if piece == "O":
            new_os = frozenset((state.os - {cell}) | {dest})
            new_xs = state.xs
        else:
            new_os = state.os
            new_xs = frozenset((state.xs - {cell}) | {dest})
            if is_x_loss(new_xs, static):
                continue

        if _is_deadlock_cached(context, new_os, new_xs, player=cell):
            continue

        next_state, next_region = _normalize_with_region(cell, new_os, new_xs, static)
        child_o_push_count = _legal_o_push_count_for(
            next_state,
            static,
            next_region,
            context.o_push_count_cache,
        )

        if goal_info(next_state, static, region=next_region) is not None:
            child_plan = None
            h = 0.0
        else:
            child_top_plans = _top_line_plans_for(
                next_state,
                static,
                region=next_region,
                target_access_penalty=context.target_access_penalty,
                top_plan_cache=context.top_plan_cache,
            )
            child_plan = child_top_plans[0] if child_top_plans else None
            if next_state not in context.plan_cache:
                context.plan_cache[next_state] = child_plan
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
            board=static,
        )
        if parent_top_plans:
            bias += min(
                _plan_specific_push_bias(
                    plan=plan,
                    push=push,
                    parent_region=region,
                    child_region=next_region,
                    board=static,
                )
                for plan in parent_top_plans
            )

        context.region_cache.setdefault(next_state, next_region)
        context.h_cache.setdefault(next_state, h)
        items.append((push, next_state, next_region, h, bias))

    items.sort(key=lambda item: (item[3] + item[4], item[3]))
    return items


def _same_piece_continuation(
    context: SearchContext,
    state: State,
    *,
    piece: str,
    cell: int,
    move: str,
) -> tuple[Push, State, frozenset[int], float, float] | None:
    piece_pushes = _push_items_for_piece_cell(
        context,
        state,
        piece=piece,
        cell=cell,
    )
    if len(piece_pushes) != 1:
        return None
    item = piece_pushes[0]
    if item[0].move != move:
        return None
    return item


def _macro_children_for(
    context: SearchContext,
    base_items: tuple[tuple[Push, State, frozenset[int], float, float], ...],
    *,
    max_extra_pushes: int = 4,
) -> list[StrategyChild]:
    macros: list[StrategyChild] = []
    static = context.static
    for push, child, _child_region, _child_h, bias in base_items:
        chain = [push]
        total_bias = bias
        current_state = child
        current_cell = _push_destination(push, static)

        for _ in range(max_extra_pushes):
            continuation = _same_piece_continuation(
                context,
                current_state,
                piece=push.piece,
                cell=current_cell,
                move=push.move,
            )
            if continuation is None:
                break
            next_push, next_state, _next_region, _next_h, next_bias = continuation
            chain.append(next_push)
            total_bias += next_bias
            current_state = next_state
            current_cell = _push_destination(next_push, static)

        if len(chain) <= 1:
            continue
        region = context.region_cache[current_state]
        h = context.h_cache[current_state]
        macro_bias = (total_bias / len(chain)) - (0.20 * (len(chain) - 1))
        macros.append((tuple(chain), current_state, region, h, macro_bias, len(chain), 0.0))
    return macros


def _policy_score_for(
    context: SearchContext,
    parent: State,
    pushes: tuple[Push, ...],
    child: State,
    child_region: frozenset[int],
    child_h: float,
    hand_bias: float,
    push_cost: int,
    *,
    hints_only: bool = False,
    raw_only: bool = False,
) -> float:
    key = (parent, pushes, child, hints_only, raw_only)
    if key in context.policy_score_cache:
        return context.policy_score_cache[key]
    try:
        from solver.push_solver.policy_features import features_for_child
        from solver.push_solver.rank_policy import default_policy
    except Exception:
        return 0.0
    policy = default_policy()
    if policy is None:
        context.policy_score_cache[key] = 0.0
        return 0.0
    if hints_only:
        score = policy.action_bonus(context.static, parent, pushes)
        context.policy_score_cache[key] = score
        return score
    features = features_for_child(
        context,
        parent,
        pushes,
        child,
        child_region,
        child_h,
        hand_bias,
        push_cost,
    )
    if raw_only:
        score = policy.raw_score(features) + policy.action_bonus(context.static, parent, pushes)
    else:
        score = policy.score(features) + policy.action_bonus(context.static, parent, pushes)
    context.policy_score_cache[key] = score
    return score


def _committed_plan_bias(
    *,
    context: SearchContext,
    parent: State,
    parent_region: frozenset[int],
    parent_h: float,
    parent_o_push_count: int,
    committed_plan: LinePlan,
    pushes: tuple[Push, ...],
    child: State,
    child_region: frozenset[int],
    child_h: float,
) -> float:
    static = context.static
    first_push = pushes[0]
    first_dest = _push_destination(first_push, static)
    important_cells = (
        set(committed_plan.line)
        | set(committed_plan.route_cells)
        | set(committed_plan.stand_cells)
        | {committed_plan.player_target}
    )
    parent_blockers = len(parent.xs & important_cells)
    child_blockers = len(child.xs & important_cells)
    bias = float(child_blockers - parent_blockers) * 2.0

    if committed_plan.player_target not in parent_region and committed_plan.player_target in child_region:
        bias -= 5.0
    elif committed_plan.player_target in parent_region and committed_plan.player_target not in child_region:
        bias += 2.0

    parent_o_pushes = parent_o_push_count
    child_o_pushes = _legal_o_push_count_for(
        child,
        static,
        child_region,
        context.o_push_count_cache,
    )
    if child_o_pushes > parent_o_pushes:
        bias -= min(child_o_pushes - parent_o_pushes, 4) * 0.75

    if child_h < parent_h:
        bias -= min(parent_h - child_h, 4.0) * 0.35

    if first_push.piece == "X":
        if first_push.cell in important_cells:
            bias -= 2.0
        if first_dest in important_cells:
            bias += 2.5
    else:
        target_deltas = []
        for target in committed_plan.o_targets:
            before = _push_distance(first_push.cell, target, static)
            after = _push_distance(first_dest, target, static)
            if before != float("inf") and after != float("inf"):
                target_deltas.append(after - before)
        if target_deltas:
            best_delta = min(target_deltas)
            if best_delta < 0:
                bias += best_delta * 1.5
            elif best_delta > 0:
                bias += min(best_delta, 3.0) * 0.75
        elif first_push.cell in important_cells:
            bias += 1.0

    if first_dest == committed_plan.player_target:
        bias += 4.0
    elif first_push.cell == committed_plan.player_target:
        bias -= 3.0

    return bias


def _is_relevant_to_commitment(
    *,
    context: SearchContext,
    parent: State,
    parent_region: frozenset[int],
    parent_h: float,
    parent_o_push_count: int,
    committed_plan: LinePlan,
    pushes: tuple[Push, ...],
    child: State,
    child_region: frozenset[int],
    child_h: float,
) -> bool:
    bias = _committed_plan_bias(
        context=context,
        parent=parent,
        parent_region=parent_region,
        parent_h=parent_h,
        parent_o_push_count=parent_o_push_count,
        committed_plan=committed_plan,
        pushes=pushes,
        child=child,
        child_region=child_region,
        child_h=child_h,
    )
    if bias < 0:
        return True
    first_push = pushes[0]
    first_dest = _push_destination(first_push, context.static)
    important_cells = (
        set(committed_plan.line)
        | set(committed_plan.route_cells)
        | set(committed_plan.stand_cells)
        | {committed_plan.player_target}
    )
    if first_push.cell in important_cells or first_dest in important_cells:
        return True
    if child_h <= parent_h:
        return True
    return False


def _important_cells_for_plan(plan: LinePlan) -> frozenset[int]:
    return frozenset(
        set(plan.line)
        | set(plan.route_cells)
        | set(plan.stand_cells)
        | {plan.player_target}
    )


def _beam_signature(
    *,
    plan: LinePlan,
    important_cells: frozenset[int],
    child: State,
    child_region: frozenset[int],
    child_h: float,
) -> tuple[frozenset[int], frozenset[int], bool, int]:
    if math.isfinite(child_h):
        h_bucket = int(child_h // 2)
    else:
        h_bucket = 1_000_000
    return (
        child.os,
        frozenset(child.xs & important_cells),
        plan.player_target in child_region,
        h_bucket,
    )


def _strategy_children_for(
    context: SearchContext,
    state: State,
    *,
    use_macros: bool,
    bias_scale: float,
    policy_weight: float,
    committed_plan: LinePlan | None = None,
    commitment_bias_scale: float = 0.0,
    relevance_filter: bool = False,
    policy_hints_only: bool = False,
    policy_raw_only: bool = False,
) -> list[StrategyChild]:
    base_items = _successors_for(context, state)
    children: list[StrategyChild] = [
        ((push,), nxt, child_region, child_h, bias, 1, 0.0)
        for push, nxt, child_region, child_h, bias in base_items
    ]
    if use_macros:
        children.extend(_macro_children_for(context, base_items))
    if policy_weight:
        children = [
            (
                pushes,
                nxt,
                child_region,
                child_h,
                bias,
                push_cost,
                _policy_score_for(
                    context,
                    state,
                    pushes,
                    nxt,
                    child_region,
                    child_h,
                    bias,
                    push_cost,
                    hints_only=policy_hints_only,
                    raw_only=policy_raw_only,
                ),
            )
            for pushes, nxt, child_region, child_h, bias, push_cost, _policy_score in children
        ]
    if committed_plan is not None and children:
        parent_region = context.region_cache[state]
        parent_h = context.h_cache[state]
        parent_o_push_count = _legal_o_push_count_for(
            state,
            context.static,
            parent_region,
            context.o_push_count_cache,
        )
        committed_children: list[StrategyChild] = []
        for pushes, nxt, child_region, child_h, bias, push_cost, policy_score in children:
            commitment_bias = _committed_plan_bias(
                context=context,
                parent=state,
                parent_region=parent_region,
                parent_h=parent_h,
                parent_o_push_count=parent_o_push_count,
                committed_plan=committed_plan,
                pushes=pushes,
                child=nxt,
                child_region=child_region,
                child_h=child_h,
            )
            committed_children.append(
                (
                    pushes,
                    nxt,
                    child_region,
                    child_h,
                    bias + (commitment_bias_scale * commitment_bias),
                    push_cost,
                    policy_score,
                )
            )
        if relevance_filter:
            filtered = [
                item
                for item in committed_children
                if _is_relevant_to_commitment(
                    context=context,
                    parent=state,
                    parent_region=parent_region,
                    parent_h=parent_h,
                    parent_o_push_count=parent_o_push_count,
                    committed_plan=committed_plan,
                    pushes=item[0],
                    child=item[1],
                    child_region=item[2],
                    child_h=item[3],
                )
            ]
            if filtered:
                filtered_keys = {
                    (item[0], item[1])
                    for item in filtered
                }
                escape_band = [
                    item
                    for item in sorted(
                        committed_children,
                        key=lambda item: (item[3] + item[4], item[3], item[5]),
                    )
                    if (item[0], item[1]) not in filtered_keys
                ][:8]
                children = filtered + escape_band
            else:
                children = committed_children
        else:
            children = committed_children
    children.sort(
        key=lambda item: (
            item[3] + (bias_scale * item[4]) - (policy_weight * item[6]),
            item[3],
            item[5],
        )
    )
    return children


def _run_strategy(
    context: SearchContext,
    *,
    config: SearchStrategyConfig,
    max_nodes: int | None,
    deadline: float | None,
) -> PushSolveResult:
    if config.kind == "committed_beam":
        return _run_committed_beam_strategy(
            context,
            config=config,
            max_nodes=max_nodes,
            deadline=deadline,
        )

    started = time.perf_counter()
    static = context.static
    start_state = context.start_state
    initial_player = context.initial_player
    nodes_expanded = 0
    peak_closed_size = 1
    counter = itertools.count()
    parents: dict[State, Parent] = {}
    start_h = context.h_cache[start_state]

    if config.kind == "rank_discrepancy":
        queue: list[tuple[float, int, int, int, State]] = []
        best_rank_cost: dict[State, tuple[int, int]] = {start_state: (0, 0)}
        heapq.heappush(queue, (0.25 * start_h, 0, 0, next(counter), start_state))
    else:
        weighted_queue: list[tuple[float, int, int, State]] = []
        g_cost = {start_state: 0}
        start_priority = (config.g_weight * 0) + (config.weight * start_h)
        heapq.heappush(weighted_queue, (start_priority, 0, next(counter), start_state))
        queue = weighted_queue

    while queue:
        if _timed_out(deadline):
            return _result(
                solved=False,
                moves=None,
                final_board=None,
                pushes=(),
                nodes_expanded=nodes_expanded,
                peak_closed_size=peak_closed_size,
                started=started,
                failure_reason="timeout",
                strategy=config.name,
            )
        if max_nodes is not None and nodes_expanded >= max_nodes:
            return _result(
                solved=False,
                moves=None,
                final_board=None,
                pushes=(),
                nodes_expanded=nodes_expanded,
                peak_closed_size=peak_closed_size,
                started=started,
                failure_reason="node_cap",
                strategy=config.name,
            )

        if config.kind == "rank_discrepancy":
            _priority, discrepancy, cost, _tie, current = heapq.heappop(queue)
            if (discrepancy, cost) != best_rank_cost.get(current):
                continue
        else:
            _priority, cost, _tie, current = heapq.heappop(queue)
            if cost != g_cost.get(current):
                continue
            discrepancy = 0
        nodes_expanded += 1

        region = context.region_cache[current]
        current_goal = goal_info(current, static, region=region)
        if current_goal is not None:
            pushes = _reconstruct_pushes(parents, current)
            moves, final_board = reconstruct_moves(
                pushes,
                current_goal,
                static,
                start_state,
                initial_player,
            )
            return _result(
                solved=True,
                moves=moves,
                final_board=final_board,
                pushes=pushes,
                nodes_expanded=nodes_expanded,
                peak_closed_size=peak_closed_size,
                started=started,
                failure_reason=None,
                strategy=config.name,
            )

        child_items = _strategy_children_for(
            context,
            current,
            use_macros=config.use_macros,
            bias_scale=config.bias_scale,
            policy_weight=config.policy_weight,
            committed_plan=config.committed_plan,
            commitment_bias_scale=config.commitment_bias_scale,
            relevance_filter=config.relevance_filter,
        )

        if config.kind == "rank_discrepancy":
            for local_rank, (pushes, nxt, child_region, child_h, _bias, push_cost, _policy_score) in enumerate(child_items):
                if nxt not in context.region_cache:
                    context.region_cache[nxt] = child_region
                    context.h_cache[nxt] = child_h
                next_cost = cost + push_cost
                next_discrepancy = discrepancy + local_rank
                next_key = (next_discrepancy, next_cost)
                if next_key >= best_rank_cost.get(nxt, (1_000_000_000, 1_000_000_000)):
                    continue
                best_rank_cost[nxt] = next_key
                parents[nxt] = Parent(previous=current, pushes=pushes)
                next_priority = next_discrepancy + (0.10 * next_cost) + (0.25 * child_h)
                heapq.heappush(
                    queue,
                    (next_priority, next_discrepancy, next_cost, next(counter), nxt),
                )
            peak_closed_size = max(peak_closed_size, len(best_rank_cost))
        else:
            for pushes, nxt, child_region, child_h, bias, push_cost, policy_score in child_items:
                if nxt not in context.region_cache:
                    context.region_cache[nxt] = child_region
                    context.h_cache[nxt] = child_h
                next_cost = cost + push_cost
                if next_cost >= g_cost.get(nxt, 1_000_000_000):
                    continue
                g_cost[nxt] = next_cost
                parents[nxt] = Parent(previous=current, pushes=pushes)
                next_priority = (
                    (config.g_weight * next_cost)
                    + (config.weight * child_h)
                    + (config.bias_scale * bias)
                    - (config.policy_weight * policy_score)
                )
                heapq.heappush(queue, (next_priority, next_cost, next(counter), nxt))
            peak_closed_size = max(peak_closed_size, len(g_cost))

    return _result(
        solved=False,
        moves=None,
        final_board=None,
        pushes=(),
        nodes_expanded=nodes_expanded,
        peak_closed_size=peak_closed_size,
        started=started,
        failure_reason="exhausted",
        strategy=config.name,
    )


def _run_committed_beam_strategy(
    context: SearchContext,
    *,
    config: SearchStrategyConfig,
    max_nodes: int | None,
    deadline: float | None,
) -> PushSolveResult:
    started = time.perf_counter()
    static = context.static
    start_state = context.start_state
    initial_player = context.initial_player
    start_region = context.region_cache[start_state]
    plans = _top_line_plans_for(
        start_state,
        static,
        region=start_region,
        target_access_penalty=context.target_access_penalty,
        top_plan_cache=context.top_plan_cache,
        limit=config.beam_plan_limit,
    )
    if not plans:
        return _result(
            solved=False,
            moves=None,
            final_board=None,
            pushes=(),
            nodes_expanded=0,
            peak_closed_size=1,
            started=started,
            failure_reason="no_line_plans",
            strategy=config.name,
        )

    nodes_expanded = 0
    peak_closed_size = 1
    counter = itertools.count()
    schedules = tuple(
        zip(config.beam_restart_widths, config.beam_restart_depths, strict=False)
    )
    if not schedules:
        schedules = ((config.beam_width, config.beam_max_depth),)

    for width, depth_limit in schedules:
        width = max(1, width)
        depth_limit = max(1, depth_limit)
        for plan in plans:
            parents: dict[State, Parent] = {}
            best_cost_by_state: dict[State, int] = {start_state: 0}
            important_cells = _important_cells_for_plan(plan)
            frontier: list[tuple[float, int, State]] = [(0.0, 0, start_state)]
            for _depth in range(depth_limit):
                if _timed_out(deadline):
                    return _result(
                        solved=False,
                        moves=None,
                        final_board=None,
                        pushes=(),
                        nodes_expanded=nodes_expanded,
                        peak_closed_size=peak_closed_size,
                        started=started,
                        failure_reason="timeout",
                        strategy=config.name,
                    )
                if max_nodes is not None and nodes_expanded >= max_nodes:
                    return _result(
                        solved=False,
                        moves=None,
                        final_board=None,
                        pushes=(),
                        nodes_expanded=nodes_expanded,
                        peak_closed_size=peak_closed_size,
                        started=started,
                        failure_reason="node_cap",
                        strategy=config.name,
                    )
                if not frontier:
                    break

                raw_candidates: list[
                    tuple[
                        float,
                        int,
                        tuple[frozenset[int], frozenset[int], bool, int],
                        State,
                        tuple[Push, ...],
                        State,
                        frozenset[int],
                        float,
                        int,
                    ]
                ] = []
                for _score, cost, current in frontier:
                    if max_nodes is not None and nodes_expanded >= max_nodes:
                        break
                    nodes_expanded += 1
                    region = context.region_cache[current]
                    current_goal = goal_info(current, static, region=region)
                    if current_goal is not None:
                        pushes = _reconstruct_pushes(parents, current)
                        moves, final_board = reconstruct_moves(
                            pushes,
                            current_goal,
                            static,
                            start_state,
                            initial_player,
                        )
                        return _result(
                            solved=True,
                            moves=moves,
                            final_board=final_board,
                            pushes=pushes,
                            nodes_expanded=nodes_expanded,
                            peak_closed_size=peak_closed_size,
                            started=started,
                            failure_reason=None,
                            strategy=config.name,
                        )

                    child_items = _strategy_children_for(
                        context,
                        current,
                        use_macros=True,
                        bias_scale=config.bias_scale,
                        policy_weight=config.policy_weight,
                        committed_plan=plan,
                        commitment_bias_scale=config.commitment_bias_scale,
                        relevance_filter=True,
                        policy_raw_only=True,
                    )
                    for pushes, nxt, child_region, child_h, bias, push_cost, policy_score in child_items:
                        next_cost = cost + push_cost
                        if next_cost >= best_cost_by_state.get(nxt, 1_000_000_000):
                            continue
                        if nxt not in context.region_cache:
                            context.region_cache[nxt] = child_region
                            context.h_cache[nxt] = child_h
                        signature = _beam_signature(
                            plan=plan,
                            important_cells=important_cells,
                            child=nxt,
                            child_region=child_region,
                            child_h=child_h,
                        )
                        base_score = (
                            child_h
                            + (1.20 * bias)
                            - (config.policy_weight * policy_score)
                            + (0.08 * push_cost)
                        )
                        raw_candidates.append(
                            (
                                base_score,
                                next(counter),
                                signature,
                                current,
                                pushes,
                                nxt,
                                child_region,
                                child_h,
                                next_cost,
                            )
                        )

                raw_candidates.sort(key=lambda item: (item[0], item[1]))
                generated_novelty_counts: dict[
                    tuple[frozenset[int], frozenset[int], bool, int],
                    int,
                ] = defaultdict(int)
                scored_candidates: list[
                    tuple[float, int, State, tuple[Push, ...], State, frozenset[int], int]
                ] = []
                for _candidate_index, (
                    base_score,
                    tie,
                    signature,
                    current,
                    pushes,
                    nxt,
                    child_region,
                    _child_h,
                    next_cost,
                ) in enumerate(raw_candidates):
                    if next_cost >= best_cost_by_state.get(nxt, 1_000_000_000):
                        continue
                    novelty_count = generated_novelty_counts[signature]
                    generated_novelty_counts[signature] += 1
                    novelty_overflow = max(
                        0,
                        novelty_count - config.beam_novelty_per_signature + 1,
                    )
                    score = base_score + (0.25 * novelty_count) + (0.75 * novelty_overflow)
                    scored_candidates.append(
                        (score, tie, current, pushes, nxt, child_region, next_cost)
                    )

                scored_candidates.sort(key=lambda item: (item[0], item[1]))
                next_frontier: list[tuple[float, int, State]] = []
                for score, _tie, current, pushes, nxt, child_region, next_cost in scored_candidates:
                    if len(next_frontier) >= width:
                        break
                    if next_cost >= best_cost_by_state.get(nxt, 1_000_000_000):
                        continue
                    best_cost_by_state[nxt] = next_cost
                    parents[nxt] = Parent(previous=current, pushes=pushes)
                    child_goal = goal_info(nxt, static, region=child_region)
                    if child_goal is not None:
                        reconstructed_pushes = _reconstruct_pushes(parents, nxt)
                        moves, final_board = reconstruct_moves(
                            reconstructed_pushes,
                            child_goal,
                            static,
                            start_state,
                            initial_player,
                        )
                        return _result(
                            solved=True,
                            moves=moves,
                            final_board=final_board,
                            pushes=reconstructed_pushes,
                            nodes_expanded=nodes_expanded,
                            peak_closed_size=max(peak_closed_size, len(best_cost_by_state)),
                            started=started,
                            failure_reason=None,
                            strategy=config.name,
                        )
                    next_frontier.append((score, next_cost, nxt))

                peak_closed_size = max(peak_closed_size, len(best_cost_by_state))
                next_frontier.sort(key=lambda item: item[0])
                frontier = next_frontier[:width]

    return _result(
        solved=False,
        moves=None,
        final_board=None,
        pushes=(),
        nodes_expanded=nodes_expanded,
        peak_closed_size=peak_closed_size,
        started=started,
        failure_reason="beam_exhausted",
        strategy=config.name,
    )
