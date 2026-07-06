"""Push-level weighted A* solver for Tic Tac Go.

This is a Sokoban-style solver: walking is handled by flood fill, and search
nodes are pushes of O/X pieces. The returned move string is still the concrete
U/D/L/R keystroke sequence expected by the rest of the app.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from solver.board_utils import normalize_board


DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)
DIRECTION_BY_MOVE = {move: (dr, dc) for move, dr, dc in DIRECTIONS}


@dataclass(frozen=True)
class StaticBoard:
    rows: int
    cols: int
    walls: frozenset[int]
    floor: frozenset[int]
    win_lines: tuple[tuple[int, int, int], ...]
    dead_cells_for_o: frozenset[int]
    push_distances: Mapping[int, Mapping[int, int]]
    push_predecessors: Mapping[int, Mapping[int, int]]
    push_stand_cells: Mapping[int, Mapping[int, int]]
    push_routes: Mapping[int, Mapping[int, tuple[int, ...]]]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def index(self, row: int, col: int) -> int:
        return (row * self.cols) + col

    def coord(self, index: int) -> tuple[int, int]:
        return divmod(index, self.cols)

    def neighbor(self, index: int, move: str) -> int | None:
        row, col = self.coord(index)
        dr, dc = DIRECTION_BY_MOVE[move]
        next_row = row + dr
        next_col = col + dc
        if not self.in_bounds(next_row, next_col):
            return None
        next_index = self.index(next_row, next_col)
        if next_index in self.walls:
            return None
        return next_index


@dataclass(frozen=True)
class State:
    player: int
    os: frozenset[int]
    xs: frozenset[int]


@dataclass(frozen=True)
class Push:
    piece: str
    cell: int
    move: str


@dataclass(frozen=True)
class Parent:
    previous: State
    push: Push


@dataclass(frozen=True)
class GoalInfo:
    line: tuple[int, int, int]
    player_target: int


@dataclass(frozen=True)
class LinePlan:
    score: float
    line: tuple[int, int, int]
    player_target: int
    route_cells: frozenset[int]
    player_target_has_x: bool
    player_target_reachable: bool


@dataclass(frozen=True)
class PushSolveResult:
    solved: bool
    moves: str | None
    final_board: tuple[tuple[str, ...], ...] | None
    pushes: tuple[Push, ...]
    nodes_expanded: int
    peak_closed_size: int
    elapsed_ms: float
    failure_reason: str | None


def parse_board(board) -> tuple[StaticBoard, State, tuple[tuple[str, ...], ...], int]:
    normalized = normalize_board(board)
    rows = len(normalized)
    cols = max((len(row) for row in normalized), default=0)
    walls: set[int] = set()
    floor: set[int] = set()
    os: set[int] = set()
    xs: set[int] = set()
    player: int | None = None

    for row_index, row in enumerate(normalized):
        for col_index, cell in enumerate(row):
            index = (row_index * cols) + col_index
            if cell == "B":
                walls.add(index)
                continue
            floor.add(index)
            if cell == "U":
                player = index
            elif cell == "O":
                os.add(index)
            elif cell == "X":
                xs.add(index)

    if player is None:
        raise ValueError("Board must contain a U player piece.")

    win_lines = _build_win_lines(rows, cols, frozenset(walls))
    static = StaticBoard(
        rows=rows,
        cols=cols,
        walls=frozenset(walls),
        floor=frozenset(floor),
        win_lines=win_lines,
        dead_cells_for_o=frozenset(),
        push_distances=MappingProxyType({}),
        push_predecessors=MappingProxyType({}),
        push_stand_cells=MappingProxyType({}),
        push_routes=MappingProxyType({}),
    )
    (
        push_distances,
        push_predecessors,
        push_stand_cells,
        push_routes,
    ) = _compute_push_distance_maps(static)
    static = StaticBoard(
        rows=static.rows,
        cols=static.cols,
        walls=static.walls,
        floor=static.floor,
        win_lines=static.win_lines,
        dead_cells_for_o=_compute_dead_cells_for_o(static, push_distances),
        push_distances=push_distances,
        push_predecessors=push_predecessors,
        push_stand_cells=push_stand_cells,
        push_routes=push_routes,
    )
    state = normalize_state(player, frozenset(os), frozenset(xs), static)
    return static, state, normalized, player


def _build_win_lines(
    rows: int,
    cols: int,
    walls: frozenset[int],
) -> tuple[tuple[int, int, int], ...]:
    lines: list[tuple[int, int, int]] = []
    for row in range(rows):
        for col in range(cols - 2):
            line = (
                (row * cols) + col,
                (row * cols) + col + 1,
                (row * cols) + col + 2,
            )
            if not any(index in walls for index in line):
                lines.append(line)
    for row in range(rows - 2):
        for col in range(cols):
            line = (
                (row * cols) + col,
                ((row + 1) * cols) + col,
                ((row + 2) * cols) + col,
            )
            if not any(index in walls for index in line):
                lines.append(line)
    return tuple(lines)


def reachable(
    player: int,
    os: frozenset[int],
    xs: frozenset[int],
    board: StaticBoard,
) -> frozenset[int]:
    blocked = board.walls | os | xs
    queue = deque([player])
    seen = {player}
    while queue:
        current = queue.popleft()
        for move, _dr, _dc in DIRECTIONS:
            nxt = board.neighbor(current, move)
            if nxt is None or nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return frozenset(seen)


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


def is_x_loss(xs: frozenset[int], board: StaticBoard) -> bool:
    return any(all(cell in xs for cell in line) for line in board.win_lines)


def _compute_push_distance_maps(
    board: StaticBoard,
) -> tuple[
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, tuple[int, ...]]],
]:
    targets = {cell for line in board.win_lines for cell in line}
    distance_maps: dict[int, Mapping[int, int]] = {}
    predecessor_maps: dict[int, Mapping[int, int]] = {}
    stand_maps: dict[int, Mapping[int, int]] = {}
    route_maps: dict[int, Mapping[int, tuple[int, ...]]] = {}
    for target in targets:
        distances = {target: 0}
        predecessors: dict[int, int] = {}
        stand_cells: dict[int, int] = {}
        queue = deque([target])

        while queue:
            current = queue.popleft()
            for _move, dr, dc in DIRECTIONS:
                row, col = board.coord(current)
                previous_row = row - dr
                previous_col = col - dc
                stand_row = row - (2 * dr)
                stand_col = col - (2 * dc)
                if not board.in_bounds(previous_row, previous_col):
                    continue
                if not board.in_bounds(stand_row, stand_col):
                    continue
                previous = board.index(previous_row, previous_col)
                stand = board.index(stand_row, stand_col)
                if previous in board.walls or stand in board.walls:
                    continue
                if previous not in distances:
                    distances[previous] = distances[current] + 1
                    predecessors[previous] = current
                    stand_cells[previous] = stand
                    queue.append(previous)
        distance_maps[target] = MappingProxyType(distances)
        predecessor_maps[target] = MappingProxyType(predecessors)
        stand_maps[target] = MappingProxyType(stand_cells)
        route_maps[target] = MappingProxyType(
            {
                cell: _build_route_cells(cell, target, predecessors, stand_cells)
                for cell in distances
            }
        )

    return (
        MappingProxyType(distance_maps),
        MappingProxyType(predecessor_maps),
        MappingProxyType(stand_maps),
        MappingProxyType(route_maps),
    )


def _build_route_cells(
    cell: int,
    target: int,
    predecessors: Mapping[int, int],
    stand_cells: Mapping[int, int],
) -> tuple[int, ...]:
    route: list[int] = [cell]
    node = cell
    while node != target:
        route.append(stand_cells[node])
        node = predecessors[node]
        route.append(node)
    return tuple(route)


def _compute_dead_cells_for_o(
    board: StaticBoard,
    push_distances: Mapping[int, Mapping[int, int]],
) -> frozenset[int]:
    reachable_cells = set()
    for distances in push_distances.values():
        reachable_cells.update(distances)
    return frozenset(board.floor - reachable_cells)


def is_deadlock(os: frozenset[int], _xs: frozenset[int], board: StaticBoard) -> bool:
    return any(cell in board.dead_cells_for_o for cell in os)


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


PENALTY_PER_BLOCKING_X: float = 1.0
PLAYER_TARGET_X_PENALTY: float = 2.0
PLAYER_TARGET_UNREACHABLE_PENALTY: float = 0.0
X_ON_LINE_PENALTY: float = 0.5
X_PUSH_BASE_BIAS: float = 0.35
O_PUSH_BASE_BIAS: float = -0.10
REACH_GAIN_BIAS: float = 0.08
O_PUSH_ACCESS_GAIN_BIAS: float = 0.80
HEURISTIC_PROGRESS_BIAS: float = 1.00
PLAN_X_CLEAR_BIAS: float = 2.00
PLAN_TARGET_REACHABLE_BIAS: float = 2.50
PLAN_TARGET_UNREACHABLE_BIAS: float = 2.00


def _route_cells(cell: int, target: int, board: StaticBoard) -> tuple[int, ...]:
    """Box-landing cells and player-stand cells along the canonical shortest
    wall-only push route from `cell` to `target` (both endpoints included)."""
    return board.push_routes.get(target, {}).get(cell, ())


def _occupancy_penalty(
    cell: int,
    target: int,
    blockers: frozenset[int],
    board: StaticBoard,
) -> float:
    base = _push_distance(cell, target, board)
    if base == math.inf:
        return math.inf
    route = _route_cells(cell, target, board)
    blocking = sum(1 for route_cell in route if route_cell in blockers)
    return base + (PENALTY_PER_BLOCKING_X * blocking)


def _best_line_plan(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int] | None = None,
    target_access_penalty: float = 0.0,
) -> LinePlan | None:
    region = region or reachable(state.player, state.os, state.xs, board)
    if len(state.os) < 2:
        return None

    o_cells = tuple(state.os)
    blockers0 = state.xs | {o_cells[1]}
    blockers1 = state.xs | {o_cells[0]}
    best_plan: LinePlan | None = None
    for line in board.win_lines:
        for player_target in line:
            targets = tuple(cell for cell in line if cell != player_target)
            if len(targets) != 2:
                continue

            plan_penalty = 0.0
            if player_target in state.xs:
                plan_penalty += PLAYER_TARGET_X_PENALTY
            if player_target not in region:
                plan_penalty += target_access_penalty
            plan_penalty += X_ON_LINE_PENALTY * sum(
                1 for cell in line if cell in state.xs
            )

            assignments = (
                (o_cells[0], targets[0], blockers0, o_cells[1], targets[1], blockers1),
                (o_cells[0], targets[1], blockers0, o_cells[1], targets[0], blockers1),
            )
            for o_a, target_a, blockers_a, o_b, target_b, blockers_b in assignments:
                cost_a = _occupancy_penalty(o_a, target_a, blockers_a, board)
                cost_b = _occupancy_penalty(o_b, target_b, blockers_b, board)
                score = cost_a + cost_b + plan_penalty
                if best_plan is not None and score >= best_plan.score:
                    continue

                route_cells = frozenset(
                    _route_cells(o_a, target_a, board)
                    + _route_cells(o_b, target_b, board)
                )
                best_plan = LinePlan(
                    score=score,
                    line=line,
                    player_target=player_target,
                    route_cells=route_cells,
                    player_target_has_x=player_target in state.xs,
                    player_target_reachable=player_target in region,
                )
    return best_plan


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
        for _move, dr, dc in DIRECTIONS:
            row, col = board.coord(cell)
            stand_row = row - dr
            stand_col = col - dc
            dest_row = row + dr
            dest_col = col + dc
            if not board.in_bounds(stand_row, stand_col):
                continue
            if not board.in_bounds(dest_row, dest_col):
                continue
            stand = board.index(stand_row, stand_col)
            dest = board.index(dest_row, dest_col)
            if stand in region and dest not in board.walls and dest not in occupied:
                count += 1
    return count


def _priority_bias(
    *,
    parent_state: State,
    parent_region: frozenset[int],
    parent_h: float,
    parent_plan: LinePlan | None,
    parent_o_push_count: int,
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

    reach_gain = max(0, len(child_region) - len(parent_region))
    if reach_gain:
        bias -= min(reach_gain, 10) * REACH_GAIN_BIAS

    child_o_push_count = _legal_o_push_count(child_state, board, child_region)
    o_push_gain = max(0, child_o_push_count - parent_o_push_count)
    if o_push_gain:
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

    return bias


def successors(
    state: State,
    board: StaticBoard,
    *,
    region: frozenset[int] | None = None,
    target_access_penalty: float = 0.0,
) -> list[tuple[Push, State, float, float]]:
    if region is None:
        region = reachable(state.player, state.os, state.xs, board)
    occupied = state.os | state.xs
    parent_plan = _best_line_plan(
        state,
        board,
        region=region,
        target_access_penalty=target_access_penalty,
    )
    parent_h = math.inf if parent_plan is None else parent_plan.score
    parent_o_push_count = _legal_o_push_count(state, board, region)
    results: list[tuple[Push, State, float, float]] = []

    for piece, cells in (("O", state.os), ("X", state.xs)):
        for cell in sorted(cells):
            for move, dr, dc in DIRECTIONS:
                row, col = board.coord(cell)
                stand_row = row - dr
                stand_col = col - dc
                dest_row = row + dr
                dest_col = col + dc
                if not board.in_bounds(stand_row, stand_col):
                    continue
                if not board.in_bounds(dest_row, dest_col):
                    continue
                stand = board.index(stand_row, stand_col)
                dest = board.index(dest_row, dest_col)
                if stand not in region:
                    continue
                if dest in board.walls or dest in occupied:
                    continue

                if piece == "O":
                    new_os = frozenset((state.os - {cell}) | {dest})
                    new_xs = state.xs
                else:
                    new_os = state.os
                    new_xs = frozenset((state.xs - {cell}) | {dest})
                    if is_x_loss(new_xs, board):
                        continue

                if is_deadlock(new_os, new_xs, board):
                    continue
                next_state, next_region = _normalize_with_region(cell, new_os, new_xs, board)
                child_plan = _best_line_plan(
                    next_state,
                    board,
                    region=next_region,
                    target_access_penalty=target_access_penalty,
                )
                h = math.inf if child_plan is None else child_plan.score
                push = Push(piece=piece, cell=cell, move=move)
                bias = _priority_bias(
                    parent_state=state,
                    parent_region=region,
                    parent_h=parent_h,
                    parent_plan=parent_plan,
                    parent_o_push_count=parent_o_push_count,
                    push=push,
                    child_state=next_state,
                    child_region=next_region,
                    child_h=h,
                    child_plan=child_plan,
                    board=board,
                )
                results.append((push, next_state, h, bias))

    results.sort(key=lambda item: (item[2] + item[3], item[2]))
    return results


def _reconstruct_pushes(
    parents: dict[State, Parent],
    goal_state: State,
) -> tuple[Push, ...]:
    pushes = []
    current = goal_state
    while current in parents:
        parent = parents[current]
        pushes.append(parent.push)
        current = parent.previous
    return tuple(reversed(pushes))


def _shortest_walk(
    start: int,
    target: int,
    os: frozenset[int],
    xs: frozenset[int],
    board: StaticBoard,
) -> str | None:
    if start == target:
        return ""
    blocked = board.walls | os | xs
    queue = deque([start])
    paths = {start: ""}
    while queue:
        current = queue.popleft()
        current_path = paths[current]
        for move, _dr, _dc in DIRECTIONS:
            nxt = board.neighbor(current, move)
            if nxt is None or nxt in blocked or nxt in paths:
                continue
            path = current_path + move
            if nxt == target:
                return path
            paths[nxt] = path
            queue.append(nxt)
    return None


def _state_to_board(
    state: State,
    actual_player: int,
    board: StaticBoard,
) -> tuple[tuple[str, ...], ...]:
    rows = []
    for row in range(board.rows):
        cells = []
        for col in range(board.cols):
            index = board.index(row, col)
            if index in board.walls:
                cells.append("B")
            elif index == actual_player:
                cells.append("U")
            elif index in state.os:
                cells.append("O")
            elif index in state.xs:
                cells.append("X")
            else:
                cells.append("")
        rows.append(tuple(cells))
    return tuple(rows)


def reconstruct_moves(
    pushes: tuple[Push, ...],
    goal: GoalInfo,
    board: StaticBoard,
    initial_state: State,
    initial_player: int,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    player = initial_player
    os = initial_state.os
    xs = initial_state.xs
    moves: list[str] = []

    for push in pushes:
        dr, dc = DIRECTION_BY_MOVE[push.move]
        row, col = board.coord(push.cell)
        stand = board.index(row - dr, col - dc)
        dest = board.index(row + dr, col + dc)
        walk = _shortest_walk(player, stand, os, xs, board)
        if walk is None:
            raise RuntimeError("Could not reconstruct walk to push stand cell.")
        moves.append(walk)
        moves.append(push.move)
        if push.piece == "O":
            if push.cell not in os:
                raise RuntimeError("Reconstruction O push no longer matches state.")
            os = frozenset((os - {push.cell}) | {dest})
        else:
            if push.cell not in xs:
                raise RuntimeError("Reconstruction X push no longer matches state.")
            xs = frozenset((xs - {push.cell}) | {dest})
        player = push.cell

    final_walk = _shortest_walk(player, goal.player_target, os, xs, board)
    if final_walk is None:
        raise RuntimeError("Could not reconstruct final walk to goal cell.")
    moves.append(final_walk)
    final_player = goal.player_target
    final_state = State(player=final_player, os=os, xs=xs)
    return "".join(moves), _state_to_board(final_state, final_player, board)


def solve(
    board,
    *,
    weight: float = 2.0,
    max_nodes: int | None = 500_000,
    timeout_seconds: float | None = 10.0,
) -> PushSolveResult:
    started = time.perf_counter()
    static, start_state, _normalized, initial_player = parse_board(board)
    nodes_expanded = 0
    peak_closed_size = 1

    if is_x_loss(start_state.xs, static):
        return PushSolveResult(
            solved=False,
            moves=None,
            final_board=None,
            pushes=(),
            nodes_expanded=1,
            peak_closed_size=1,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            failure_reason="x_loss",
        )

    start_goal = goal_info(start_state, static)
    if start_goal is not None:
        moves, final_board = reconstruct_moves((), start_goal, static, start_state, initial_player)
        return PushSolveResult(
            solved=True,
            moves=moves,
            final_board=final_board,
            pushes=(),
            nodes_expanded=1,
            peak_closed_size=1,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            failure_reason=None,
        )

    if is_deadlock(start_state.os, start_state.xs, static):
        return PushSolveResult(
            solved=False,
            moves=None,
            final_board=None,
            pushes=(),
            nodes_expanded=1,
            peak_closed_size=1,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            failure_reason="deadlock",
        )

    queue: list[tuple[float, int, int, State]] = []
    counter = itertools.count()
    g_cost = {start_state: 0}
    parents: dict[State, Parent] = {}
    start_region = reachable(start_state.player, start_state.os, start_state.xs, static)
    target_access_penalty = (
        3.0
        if len(start_region) <= 2
        and _legal_o_push_count(start_state, static, start_region) > 0
        else 0.0
    )
    start_h = heuristic(
        start_state,
        static,
        region=start_region,
        target_access_penalty=target_access_penalty,
    )
    heapq.heappush(queue, (start_h * weight, 0, next(counter), start_state))

    while queue:
        if timeout_seconds is not None and (time.perf_counter() - started) >= timeout_seconds:
            return PushSolveResult(
                solved=False,
                moves=None,
                final_board=None,
                pushes=(),
                nodes_expanded=nodes_expanded,
                peak_closed_size=peak_closed_size,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                failure_reason="timeout",
            )
        if max_nodes is not None and nodes_expanded >= max_nodes:
            return PushSolveResult(
                solved=False,
                moves=None,
                final_board=None,
                pushes=(),
                nodes_expanded=nodes_expanded,
                peak_closed_size=peak_closed_size,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                failure_reason="node_cap",
            )

        _priority, cost, _tie, current = heapq.heappop(queue)
        if cost != g_cost.get(current):
            continue
        nodes_expanded += 1

        region = reachable(current.player, current.os, current.xs, static)
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
            return PushSolveResult(
                solved=True,
                moves=moves,
                final_board=final_board,
                pushes=pushes,
                nodes_expanded=nodes_expanded,
                peak_closed_size=peak_closed_size,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                failure_reason=None,
            )

        for push, nxt, h, bias in successors(
            current,
            static,
            region=region,
            target_access_penalty=target_access_penalty,
        ):
            next_cost = cost + 1
            if next_cost >= g_cost.get(nxt, 1_000_000_000):
                continue
            g_cost[nxt] = next_cost
            parents[nxt] = Parent(previous=current, push=push)
            next_priority = next_cost + (weight * h) + bias
            heapq.heappush(queue, (next_priority, next_cost, next(counter), nxt))

        peak_closed_size = max(peak_closed_size, len(g_cost))

    return PushSolveResult(
        solved=False,
        moves=None,
        final_board=None,
        pushes=(),
        nodes_expanded=nodes_expanded,
        peak_closed_size=peak_closed_size,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        failure_reason="exhausted",
    )
