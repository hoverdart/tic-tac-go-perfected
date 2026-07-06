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
    )
    push_distances = _compute_push_distance_maps(static)
    static = StaticBoard(
        rows=static.rows,
        cols=static.cols,
        walls=static.walls,
        floor=static.floor,
        win_lines=static.win_lines,
        dead_cells_for_o=_compute_dead_cells_for_o(static, push_distances),
        push_distances=push_distances,
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


def normalize_state(
    player: int,
    os: frozenset[int],
    xs: frozenset[int],
    board: StaticBoard,
) -> State:
    region = reachable(player, os, xs, board)
    return State(player=min(region), os=os, xs=xs)


def is_x_loss(xs: frozenset[int], board: StaticBoard) -> bool:
    return any(all(cell in xs for cell in line) for line in board.win_lines)


def _compute_push_distance_maps(
    board: StaticBoard,
) -> Mapping[int, Mapping[int, int]]:
    targets = {cell for line in board.win_lines for cell in line}
    maps: dict[int, Mapping[int, int]] = {}
    for target in targets:
        distances = {target: 0}
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
                    queue.append(previous)
        maps[target] = MappingProxyType(distances)

    return MappingProxyType(maps)


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


def heuristic(state: State, board: StaticBoard) -> float:
    if goal_info(state, board) is not None:
        return 0.0
    if len(state.os) < 2:
        return math.inf

    o_cells = tuple(state.os)
    best = math.inf
    for line in board.win_lines:
        for player_target in line:
            targets = tuple(cell for cell in line if cell != player_target)
            if len(targets) != 2:
                continue
            cost_a = _push_distance(o_cells[0], targets[0], board) + _push_distance(
                o_cells[1], targets[1], board
            )
            cost_b = _push_distance(o_cells[0], targets[1], board) + _push_distance(
                o_cells[1], targets[0], board
            )
            best = min(best, cost_a, cost_b)
    return best


def _push_distance(cell: int, target: int, board: StaticBoard) -> float:
    return board.push_distances.get(target, {}).get(cell, math.inf)


def successors(state: State, board: StaticBoard) -> list[tuple[Push, State]]:
    region = reachable(state.player, state.os, state.xs, board)
    occupied = state.os | state.xs
    results: list[tuple[Push, State]] = []

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
                next_state = normalize_state(cell, new_os, new_xs, board)
                results.append((Push(piece=piece, cell=cell, move=move), next_state))

    results.sort(key=lambda item: heuristic(item[1], board))
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
    heapq.heappush(queue, (heuristic(start_state, static) * weight, 0, next(counter), start_state))

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

        for push, nxt in successors(current, static):
            next_cost = cost + 1
            if next_cost >= g_cost.get(nxt, 1_000_000_000):
                continue
            g_cost[nxt] = next_cost
            parents[nxt] = Parent(previous=current, push=push)
            next_priority = next_cost + (weight * heuristic(nxt, static))
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
