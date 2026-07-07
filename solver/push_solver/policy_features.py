"""Feature extraction for push-level successor ranking."""

from __future__ import annotations

import math
from typing import Any

from solver.push_solver import core


def _finite_delta(after: float, before: float) -> float:
    if math.isfinite(after) and math.isfinite(before):
        return after - before
    if math.isfinite(after):
        return -20.0
    if math.isfinite(before):
        return 20.0
    return 0.0


def _norm_coord(cell: int, board: core.StaticBoard) -> tuple[float, float]:
    row, col = board.coord(cell)
    return (
        row / max(1, board.rows - 1),
        col / max(1, board.cols - 1),
    )


def features_for_child(
    context: Any,
    parent: core.State,
    pushes: tuple[core.Push, ...],
    child: core.State,
    child_region: frozenset[int],
    child_h: float,
    hand_bias: float,
    push_cost: int,
) -> dict[str, float]:
    """Return scalar features for ranking one legal child.

    The feature function only observes states already produced by the classical
    successor generator. It does not decide legality or pruning.
    """
    board = context.static
    parent_region = context.region_cache[parent]
    parent_h = context.h_cache[parent]
    push = pushes[0]
    dest = core._push_destination(push, board)
    source_row, source_col = _norm_coord(push.cell, board)
    dest_row, dest_col = _norm_coord(dest, board)

    parent_o_pushes = core._legal_o_push_count_for(
        parent,
        board,
        parent_region,
        context.o_push_count_cache,
    )
    child_o_pushes = core._legal_o_push_count_for(
        child,
        board,
        child_region,
        context.o_push_count_cache,
    )
    parent_plans = core._top_line_plans_for(
        parent,
        board,
        region=parent_region,
        target_access_penalty=context.target_access_penalty,
        top_plan_cache=context.top_plan_cache,
    )
    child_plans = core._top_line_plans_for(
        child,
        board,
        region=child_region,
        target_access_penalty=context.target_access_penalty,
        top_plan_cache=context.top_plan_cache,
    )

    parent_best = parent_plans[0] if parent_plans else None
    child_best = child_plans[0] if child_plans else None
    top_lines = {plan.line for plan in parent_plans}
    top_player_targets = {plan.player_target for plan in parent_plans}
    top_route_cells = set().union(*(set(plan.route_cells) for plan in parent_plans)) if parent_plans else set()
    top_stand_cells = set().union(*(set(plan.stand_cells) for plan in parent_plans)) if parent_plans else set()
    top_line_cells = set().union(*(set(plan.line) for plan in parent_plans)) if parent_plans else set()

    improved_plans = 0
    harmed_plans = 0
    for parent_plan in parent_plans:
        matching_child_scores = [
            plan.score
            for plan in child_plans
            if plan.line == parent_plan.line
            and plan.player_target == parent_plan.player_target
        ]
        if not matching_child_scores:
            continue
        best_score = min(matching_child_scores)
        if best_score < parent_plan.score:
            improved_plans += 1
        elif best_score > parent_plan.score:
            harmed_plans += 1

    clears_player_target = push.cell in top_player_targets and dest not in top_player_targets
    enters_player_target = dest in top_player_targets
    clears_line = push.cell in top_line_cells and dest not in top_line_cells
    enters_line = dest in top_line_cells
    clears_route = push.cell in top_route_cells and dest not in top_route_cells
    enters_route = dest in top_route_cells
    clears_stand = push.cell in top_stand_cells and dest not in top_stand_cells
    enters_stand = dest in top_stand_cells

    x_threat_before = sum(
        1 for line in board.win_lines if sum(1 for cell in line if cell in parent.xs) == 2
    )
    x_threat_after = sum(
        1 for line in board.win_lines if sum(1 for cell in line if cell in child.xs) == 2
    )

    assigned_progress = 0.0
    if push.piece == "O" and parent_best is not None:
        targets = {o_cell: target for o_cell, target in parent_best.o_assignment}
        target = targets.get(push.cell)
        if target is not None:
            before = core._push_distance(push.cell, target, board)
            after = core._push_distance(dest, target, board)
            assigned_progress = _finite_delta(before, after)

    features = {
        "bias": hand_bias,
        "hand_score": -(child_h + hand_bias),
        "piece_o": 1.0 if push.piece == "O" else 0.0,
        "piece_x": 1.0 if push.piece == "X" else 0.0,
        "dir_u": 1.0 if push.move == "U" else 0.0,
        "dir_d": 1.0 if push.move == "D" else 0.0,
        "dir_l": 1.0 if push.move == "L" else 0.0,
        "dir_r": 1.0 if push.move == "R" else 0.0,
        "source_row": source_row,
        "source_col": source_col,
        "dest_row": dest_row,
        "dest_col": dest_col,
        "source_degree": len(board.adjacency[push.cell]) / 4.0,
        "dest_degree": len(board.adjacency[dest]) / 4.0,
        "parent_h": min(parent_h, 50.0) if math.isfinite(parent_h) else 50.0,
        "child_h": min(child_h, 50.0) if math.isfinite(child_h) else 50.0,
        "h_delta": _finite_delta(child_h, parent_h),
        "region_delta": (len(child_region) - len(parent_region)) / max(1, len(board.floor)),
        "legal_o_push_delta": (child_o_pushes - parent_o_pushes) / 8.0,
        "push_cost": float(push_cost),
        "macro_extra": float(max(0, push_cost - 1)),
        "source_on_top_line": 1.0 if push.cell in top_line_cells else 0.0,
        "dest_on_top_line": 1.0 if dest in top_line_cells else 0.0,
        "clears_player_target": 1.0 if clears_player_target else 0.0,
        "enters_player_target": 1.0 if enters_player_target else 0.0,
        "clears_line": 1.0 if clears_line else 0.0,
        "enters_line": 1.0 if enters_line else 0.0,
        "clears_route": 1.0 if clears_route else 0.0,
        "enters_route": 1.0 if enters_route else 0.0,
        "clears_stand": 1.0 if clears_stand else 0.0,
        "enters_stand": 1.0 if enters_stand else 0.0,
        "assigned_o_progress": assigned_progress,
        "x_threat_delta": float(x_threat_after - x_threat_before),
        "plans_improved": improved_plans / 8.0,
        "plans_harmed": harmed_plans / 8.0,
        "same_best_line": 1.0
        if parent_best is not None
        and child_best is not None
        and parent_best.line == child_best.line
        else 0.0,
        "top_plan_count": len(top_lines) / 8.0,
    }
    return features

