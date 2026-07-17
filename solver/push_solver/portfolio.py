"""Portfolio orchestration for push-level search strategies."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from solver.push_solver.deadlocks import is_deadlock, is_x_loss
from solver.push_solver.heuristics import goal_info
from solver.push_solver.heuristics import _top_line_plans_for
from solver.push_solver.models import (
    PushSolveResult,
    SearchAttempt,
    SearchStrategyConfig,
    State,
    StaticBoard,
)
from solver.push_solver.reconstruction import reconstruct_moves
from solver.push_solver.search import _build_search_context, _run_strategy, _timed_out
from solver.push_solver.state import parse_board


def _x_cluster_stats(xs: frozenset[int], static: StaticBoard) -> tuple[int, int, int]:
    if not xs:
        return 0, 0, 0
    seen: set[int] = set()
    component_count = 0
    largest_component = 0
    adjacency_edges = 0
    for cell in xs:
        adjacency_edges += sum(1 for nxt in static.adjacency[cell] if nxt in xs)
        if cell in seen:
            continue
        component_count += 1
        stack = [cell]
        seen.add(cell)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in static.adjacency[current]:
                if nxt in xs and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest_component = max(largest_component, size)
    return component_count, largest_component, adjacency_edges // 2


def _portfolio_context_profile(context) -> str | None:
    static = context.static
    state = context.start_state
    x_count = len(state.xs)
    floor_count = max(1, len(static.floor))
    wall_count = len(static.walls)
    threat_lines = sum(
        1 for line in static.win_lines if sum(1 for cell in line if cell in state.xs) == 2
    )
    _component_count, _largest_component, adjacency_edges = _x_cluster_stats(
        state.xs,
        static,
    )
    x_density = x_count / floor_count

    if x_count >= 18 and x_density >= 0.30 and adjacency_edges <= 1 and threat_lines <= 2:
        return "rank_first"
    if (
        x_count >= 18
        and x_density >= 0.30
        and wall_count <= 8
        and threat_lines >= 12
        and adjacency_edges >= 6
    ):
        return "policy_rank_first"
    return None


def _attempt_from_result(result: PushSolveResult) -> SearchAttempt:
    return SearchAttempt(
        strategy=result.strategy or "unknown",
        solved=result.solved,
        nodes_expanded=result.nodes_expanded,
        peak_closed_size=result.peak_closed_size,
        elapsed_ms=result.elapsed_ms,
        failure_reason=result.failure_reason,
    )


def _result(
    *,
    solved: bool,
    moves: str | None,
    final_board: tuple[tuple[str, ...], ...] | None,
    pushes: tuple,
    nodes_expanded: int,
    peak_closed_size: int,
    started: float,
    failure_reason: str | None,
    strategy: str | None,
    attempts: tuple[SearchAttempt, ...] = (),
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
        attempts=attempts,
    )


def _precheck_result(
    static: StaticBoard,
    start_state: State,
    initial_player: int,
    *,
    started: float,
) -> PushSolveResult | None:
    if is_x_loss(start_state.xs, static):
        return _result(
            solved=False,
            moves=None,
            final_board=None,
            pushes=(),
            nodes_expanded=1,
            peak_closed_size=1,
            started=started,
            failure_reason="x_loss",
            strategy="precheck",
        )

    start_goal = goal_info(start_state, static)
    if start_goal is not None:
        moves, final_board = reconstruct_moves(
            (),
            start_goal,
            static,
            start_state,
            initial_player,
        )
        return _result(
            solved=True,
            moves=moves,
            final_board=final_board,
            pushes=(),
            nodes_expanded=1,
            peak_closed_size=1,
            started=started,
            failure_reason=None,
            strategy="precheck",
        )

    if is_deadlock(start_state.os, start_state.xs, static, player=start_state.player):
        return _result(
            solved=False,
            moves=None,
            final_board=None,
            pushes=(),
            nodes_expanded=1,
            peak_closed_size=1,
            started=started,
            failure_reason="deadlock",
            strategy="precheck",
        )
    return None


def _line_committed_configs(
    context,
    *,
    weight: float,
    limit: int = 4,
    fraction: float = 0.03,
) -> tuple[tuple[SearchStrategyConfig, float], ...]:
    start_region = context.region_cache[context.start_state]
    plans = _top_line_plans_for(
        context.start_state,
        context.static,
        region=start_region,
        target_access_penalty=context.target_access_penalty,
        top_plan_cache=context.top_plan_cache,
        limit=limit,
    )[:limit]
    return tuple(
        (
            SearchStrategyConfig(
                name=f"line_commit_{index}",
                kind="rank_discrepancy",
                weight=weight,
                g_weight=0.35,
                bias_scale=1.0,
                use_macros=False,
                committed_plan=plan,
                commitment_bias_scale=1.0,
                relevance_filter=True,
            ),
            fraction,
        )
        for index, plan in enumerate(plans, start=1)
    )


def _portfolio_configs(
    weight: float,
    context=None,
) -> tuple[tuple[SearchStrategyConfig, float], ...]:
    committed = (
        _line_committed_configs(context, weight=weight, limit=1, fraction=0.005)
        if context is not None
        else ()
    )
    base_configs = (
        (
            SearchStrategyConfig(
                name="v1_weighted",
                kind="weighted",
                weight=weight,
                g_weight=1.0,
                bias_scale=1.0,
            ),
            0.08,
        ),
        (
            SearchStrategyConfig(
                name="greedy_low_g",
                kind="weighted",
                weight=max(weight, 2.2),
                g_weight=0.25,
                bias_scale=1.0,
            ),
            0.03,
        ),
        (
            SearchStrategyConfig(
                name="greedy_bias",
                kind="weighted",
                weight=max(weight, 2.5),
                g_weight=0.15,
                bias_scale=1.75,
            ),
            0.03,
        ),
        *committed,
        (
            SearchStrategyConfig(
                name="macro_greedy",
                kind="weighted",
                weight=max(weight, 2.5),
                g_weight=0.20,
                bias_scale=1.50,
                use_macros=True,
            ),
            0.09,
        ),
        (
            SearchStrategyConfig(
                name="committed_beam_recovery",
                kind="committed_beam",
                bias_scale=1.20,
                policy_weight=2.00,
                commitment_bias_scale=1.50,
                relevance_filter=True,
            ),
            0.45,
        ),
        (
            SearchStrategyConfig(
                name="rank_discrepancy",
                kind="rank_discrepancy",
                weight=weight,
                g_weight=1.0,
                bias_scale=1.25,
            ),
            0.35,
        ),
        (
            SearchStrategyConfig(
                name="policy_rank_discrepancy",
                kind="rank_discrepancy",
                weight=weight,
                g_weight=1.0,
                bias_scale=1.0,
                use_macros=True,
                policy_weight=2.00,
            ),
            0.55,
        ),
        (
            SearchStrategyConfig(
                name="committed_beam",
                kind="committed_beam",
                bias_scale=1.20,
                policy_weight=2.00,
                commitment_bias_scale=1.50,
                relevance_filter=True,
            ),
            1.00,
        ),
    )
    if context is None:
        return base_configs

    profile = _portfolio_context_profile(context)
    if profile == "rank_first":
        return (
            (
                SearchStrategyConfig(
                    name="rank_recovery_first",
                    kind="rank_discrepancy",
                    weight=weight,
                    g_weight=1.0,
                    bias_scale=1.25,
                ),
                0.45,
            ),
            *base_configs,
        )
    if profile == "policy_rank_first":
        return (
            (
                SearchStrategyConfig(
                    name="policy_rank_recovery_first",
                    kind="rank_discrepancy",
                    weight=weight,
                    g_weight=1.0,
                    bias_scale=1.0,
                    use_macros=True,
                    policy_weight=2.00,
                ),
                0.55,
            ),
            *base_configs,
        )
    return base_configs


def _attempt_budget(
    *,
    remaining: int | None,
    fraction: float,
    is_last: bool,
) -> int | None:
    if remaining is None:
        return None
    if is_last:
        return remaining
    return max(1, min(remaining, math.ceil(remaining * fraction)))


def solve_v1(
    board,
    *,
    weight: float = 2.0,
    max_nodes: int | None = 500_000,
    timeout_seconds: float | None = 10.0,
) -> PushSolveResult:
    started = time.perf_counter()
    static, start_state, _normalized, initial_player = parse_board(board)
    precheck = _precheck_result(
        static,
        start_state,
        initial_player,
        started=started,
    )
    if precheck is not None:
        return replace(precheck, strategy="v1_weighted")

    deadline = (
        started + timeout_seconds
        if timeout_seconds is not None
        else None
    )
    context = _build_search_context(static, start_state, initial_player)
    return _run_strategy(
        context,
        config=SearchStrategyConfig(
            name="v1_weighted",
            kind="weighted",
            weight=weight,
            g_weight=1.0,
            bias_scale=1.0,
        ),
        max_nodes=max_nodes,
        deadline=deadline,
    )


def solve(
    board,
    *,
    weight: float = 2.0,
    max_nodes: int | None = 500_000,
    timeout_seconds: float | None = 10.0,
) -> PushSolveResult:
    started = time.perf_counter()
    static, start_state, _normalized, initial_player = parse_board(board)
    precheck = _precheck_result(
        static,
        start_state,
        initial_player,
        started=started,
    )
    if precheck is not None:
        return precheck

    deadline = (
        started + timeout_seconds
        if timeout_seconds is not None
        else None
    )
    attempts: list[SearchAttempt] = []
    total_nodes = 0
    peak_closed_size = 1
    context = _build_search_context(static, start_state, initial_player)
    configs = _portfolio_configs(weight, context=context)

    for index, (config, fraction) in enumerate(configs):
        if _timed_out(deadline):
            break
        remaining_nodes = None if max_nodes is None else max_nodes - total_nodes
        if remaining_nodes is not None and remaining_nodes <= 0:
            break

        is_last = index == len(configs) - 1
        strategy_max_nodes = _attempt_budget(
            remaining=remaining_nodes,
            fraction=fraction,
            is_last=is_last,
        )
        if deadline is None:
            strategy_deadline = None
        elif is_last:
            strategy_deadline = deadline
        else:
            remaining_seconds = max(0.0, deadline - time.perf_counter())
            strategy_deadline = time.perf_counter() + (remaining_seconds * fraction)

        result = _run_strategy(
            context,
            config=config,
            max_nodes=strategy_max_nodes,
            deadline=strategy_deadline,
        )
        attempts.append(_attempt_from_result(result))
        total_nodes += result.nodes_expanded
        peak_closed_size = max(peak_closed_size, result.peak_closed_size)

        if result.solved:
            from solver.push_solver.verify import verify_solution

            verification = verify_solution(board, result.moves)
            if verification.ok:
                return PushSolveResult(
                    solved=True,
                    moves=result.moves,
                    final_board=result.final_board,
                    pushes=result.pushes,
                    nodes_expanded=total_nodes,
                    peak_closed_size=peak_closed_size,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    failure_reason=None,
                    strategy=config.name,
                    attempts=tuple(attempts),
                )
            attempts[-1] = SearchAttempt(
                strategy=config.name,
                solved=False,
                nodes_expanded=result.nodes_expanded,
                peak_closed_size=result.peak_closed_size,
                elapsed_ms=result.elapsed_ms,
                failure_reason=f"invalid_solution:{verification.error}",
            )

    if _timed_out(deadline):
        failure_reason = "timeout"
    elif max_nodes is not None and total_nodes >= max_nodes:
        failure_reason = "node_cap"
    elif attempts:
        failure_reason = attempts[-1].failure_reason or "portfolio_exhausted"
    else:
        failure_reason = "portfolio_exhausted"

    return PushSolveResult(
        solved=False,
        moves=None,
        final_board=None,
        pushes=(),
        nodes_expanded=total_nodes,
        peak_closed_size=peak_closed_size,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        failure_reason=failure_reason,
        strategy=None,
        attempts=tuple(attempts),
    )
