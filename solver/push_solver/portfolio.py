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
from solver.push_solver.search import (
    _build_search_context,
    _successors_for,
    _run_strategy,
    _timed_out,
)
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
    start_region_size = len(context.region_cache[context.start_state])

    if (
        14 <= x_count <= 17
        and x_density >= 0.25
        and wall_count <= 6
        and threat_lines >= 8
        and adjacency_edges <= 5
        and start_region_size <= 2
    ):
        return "greedy_first"
    if (
        12 <= x_count <= 14
        and 12 <= wall_count <= 14
        and 4 <= threat_lines <= 6
        and adjacency_edges <= 4
        and 5 <= start_region_size <= 10
    ):
        return "v1_deep_first"
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


def _has_start_ranker_hint(context) -> bool:
    """Whether the loaded ranker knows a legal first push for this position.

    Exact state-action hints come only from verified solution paths.  Give them
    a short early recovery window, while retaining the normal portfolio if that
    guided attempt cannot finish.
    """
    try:
        from solver.push_solver.rank_policy import default_policy

        policy = default_policy()
    except Exception:
        return False
    if policy is None:
        return False
    return any(
        policy.priority_action_bonus(context.static, context.start_state, (push,))
        > 0.0
        for push, *_rest in _successors_for(context, context.start_state)
    )


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

    if _has_start_ranker_hint(context):
        return (
            (
                SearchStrategyConfig(
                    name="learned_hint_recovery_first",
                    kind="rank_discrepancy",
                    weight=weight,
                    g_weight=1.0,
                    bias_scale=1.0,
                    use_macros=True,
                    policy_weight=2.0,
                ),
                0.15,
            ),
            *base_configs,
        )

    profile = _portfolio_context_profile(context)
    if profile == "v1_deep_first":
        return (
            (
                SearchStrategyConfig(
                    name="v1_deep_recovery_first",
                    kind="weighted",
                    weight=weight,
                    g_weight=1.0,
                    bias_scale=1.0,
                ),
                1.00,
            ),
            *base_configs,
        )
    if profile == "greedy_first":
        return (
            (
                SearchStrategyConfig(
                    name="greedy_low_g_recovery_first",
                    kind="weighted",
                    weight=max(weight, 2.2),
                    g_weight=0.25,
                    bias_scale=1.0,
                ),
                0.20,
            ),
            *base_configs,
        )
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


def solve_v2(
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


def solve(
    board,
    *,
    weight: float = 2.0,
    max_nodes: int | None = 500_000,
    timeout_seconds: float | None = 10.0,
    quality_timeout_seconds: float | None = 10.0,
) -> PushSolveResult:
    """Run coverage-preserving V3, then improve its V2 incumbent by keystrokes.

    Coverage is a hard invariant: V2 receives the complete caller-provided time
    and node budgets first.  V3 only runs after V2 has already produced a
    verified solution, uses nodes left over from that solved attempt, and always
    falls back to the incumbent if quality search times out, fails, or returns
    an invalid path.
    """
    started = time.perf_counter()
    baseline = solve_v2(
        board,
        weight=weight,
        max_nodes=max_nodes,
        timeout_seconds=timeout_seconds,
    )
    baseline_keystrokes = (
        len(baseline.moves)
        if baseline.solved and baseline.moves is not None
        else None
    )

    if (
        not baseline.solved
        or baseline.moves is None
        or baseline.final_board is None
        or not baseline.moves
        or quality_timeout_seconds == 0
    ):
        return replace(
            baseline,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            baseline_keystrokes=baseline_keystrokes,
        )

    remaining_nodes = (
        None
        if max_nodes is None
        else max(0, max_nodes - baseline.nodes_expanded)
    )
    if remaining_nodes == 0:
        return replace(
            baseline,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            baseline_keystrokes=baseline_keystrokes,
        )

    coverage_deadline = (
        None if timeout_seconds is None else started + timeout_seconds
    )
    quality_deadline = (
        None
        if quality_timeout_seconds is None
        else started + quality_timeout_seconds
    )
    if coverage_deadline is not None:
        quality_deadline = (
            coverage_deadline
            if quality_deadline is None
            else min(quality_deadline, coverage_deadline)
        )
    if quality_deadline is not None and time.perf_counter() >= quality_deadline:
        return replace(
            baseline,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            baseline_keystrokes=baseline_keystrokes,
        )

    quality_started = time.perf_counter()
    try:
        from solver.push_solver.optimizer import improve_solution

        static, start_state, _normalized, initial_player = parse_board(board)
        quality = improve_solution(
            static=static,
            start_state=start_state,
            initial_player=initial_player,
            incumbent_moves=baseline.moves,
            incumbent_final_board=baseline.final_board,
            incumbent_pushes=baseline.pushes,
            max_nodes=remaining_nodes,
            deadline=quality_deadline,
            weight=max(4.0, weight),
        )
    except Exception as exc:
        quality_attempt = SearchAttempt(
            strategy="v3_keystroke_anytime",
            solved=False,
            nodes_expanded=0,
            peak_closed_size=0,
            elapsed_ms=(time.perf_counter() - quality_started) * 1000,
            failure_reason=f"optimizer_error:{type(exc).__name__}",
        )
        return replace(
            baseline,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            attempts=tuple((*baseline.attempts, quality_attempt)),
            baseline_keystrokes=baseline_keystrokes,
        )

    quality_attempt = SearchAttempt(
        strategy="v3_keystroke_anytime",
        solved=quality.improved,
        nodes_expanded=quality.nodes_expanded,
        peak_closed_size=quality.peak_closed_size,
        elapsed_ms=quality.elapsed_ms,
        failure_reason=None if quality.improved else quality.termination_reason,
    )
    total_nodes = baseline.nodes_expanded + quality.nodes_expanded
    peak_closed_size = max(baseline.peak_closed_size, quality.peak_closed_size)
    attempts = tuple((*baseline.attempts, quality_attempt))

    if quality.improved:
        from solver.push_solver.verify import verify_solution

        verification = verify_solution(board, quality.moves)
        if verification.ok and len(quality.moves) < len(baseline.moves):
            return PushSolveResult(
                solved=True,
                moves=quality.moves,
                final_board=verification.final_board,
                pushes=quality.pushes,
                nodes_expanded=total_nodes,
                peak_closed_size=peak_closed_size,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                failure_reason=None,
                strategy="v3_keystroke_anytime",
                attempts=attempts,
                baseline_keystrokes=baseline_keystrokes,
                quality_improved=True,
                quality_nodes_expanded=quality.nodes_expanded,
            )
        attempts = tuple(
            (
                *baseline.attempts,
                replace(
                    quality_attempt,
                    solved=False,
                    failure_reason=(
                        f"invalid_solution:{verification.error}"
                        if not verification.ok
                        else "not_shorter"
                    ),
                ),
            )
        )

    return replace(
        baseline,
        nodes_expanded=total_nodes,
        peak_closed_size=peak_closed_size,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        attempts=attempts,
        baseline_keystrokes=baseline_keystrokes,
        quality_nodes_expanded=quality.nodes_expanded,
    )
