"""Board parsing, static push maps, and push-state normalization."""

from __future__ import annotations

import itertools
from collections import deque
from types import MappingProxyType
from typing import Mapping

from solver.board_utils import normalize_board
from solver.push_solver.models import (
    DIRECTION_BY_MOVE,
    DIRECTIONS,
    State,
    StaticBoard,
)


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

    adjacency_list = []
    transitions_list = []
    for i in range(rows * cols):
        if i in walls:
            adjacency_list.append(())
            transitions_list.append(())
            continue

        r, c = divmod(i, cols)

        valid_neighbors = []
        for _, dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                ni = nr * cols + nc
                if ni not in walls:
                    valid_neighbors.append(ni)
        adjacency_list.append(tuple(valid_neighbors))

        cell_transitions = []
        for move, dr, dc in DIRECTIONS:
            sr, sc = r - dr, c - dc
            dr_dest, dc_dest = r + dr, c + dc
            if 0 <= sr < rows and 0 <= sc < cols and 0 <= dr_dest < rows and 0 <= dc_dest < cols:
                stand = sr * cols + sc
                dest = dr_dest * cols + dc_dest
                if dest not in walls and stand not in walls:
                    cell_transitions.append((move, stand, dest))
        transitions_list.append(tuple(cell_transitions))

    static_temp = StaticBoard(
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
        push_stand_routes=MappingProxyType({}),
        adjacency=tuple(adjacency_list),
        push_transitions=tuple(transitions_list),
        push_route_sets=MappingProxyType({}),
        push_stand_route_sets=MappingProxyType({}),
        reachable_o_pairs=frozenset(),
    )

    (
        push_distances,
        push_predecessors,
        push_stand_cells,
        push_routes,
        push_stand_routes,
        push_route_sets,
        push_stand_route_sets,
    ) = _compute_push_distance_maps(static_temp)

    static = StaticBoard(
        rows=static_temp.rows,
        cols=static_temp.cols,
        walls=static_temp.walls,
        floor=static_temp.floor,
        win_lines=static_temp.win_lines,
        dead_cells_for_o=_compute_dead_cells_for_o(static_temp, push_distances),
        push_distances=push_distances,
        push_predecessors=push_predecessors,
        push_stand_cells=push_stand_cells,
        push_routes=push_routes,
        push_stand_routes=push_stand_routes,
        adjacency=static_temp.adjacency,
        push_transitions=static_temp.push_transitions,
        push_route_sets=push_route_sets,
        push_stand_route_sets=push_stand_route_sets,
        reachable_o_pairs=_compute_reachable_o_pairs(static_temp, push_distances),
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
    blocked = os | xs
    queue = [player]
    seen = {player}
    adj = board.adjacency
    for current in queue:
        for nxt in adj[current]:
            if nxt not in blocked and nxt not in seen:
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


def _compute_push_distance_maps(
    board: StaticBoard,
) -> tuple[
    Mapping[tuple[int, int], int],
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, tuple[int, ...]]],
    Mapping[int, Mapping[int, tuple[int, ...]]],
    Mapping[tuple[int, int], frozenset[int]],
    Mapping[tuple[int, int], frozenset[int]],
]:
    """Build push-distance/route lookups.

    ``push_distances``/``push_route_sets``/``push_stand_route_sets`` are kept
    flat (keyed by ``(target, cell)``) rather than nested per-target dicts:
    they sit on the line-plan hot path, and a single tuple-keyed lookup is
    cheaper per call than two chained dict lookups plus a default-dict
    allocation for missing targets.
    """
    targets = {cell for line in board.win_lines for cell in line}
    flat_distances: dict[tuple[int, int], int] = {}
    predecessor_maps: dict[int, Mapping[int, int]] = {}
    stand_maps: dict[int, Mapping[int, int]] = {}
    route_maps: dict[int, Mapping[int, tuple[int, ...]]] = {}
    stand_route_maps: dict[int, Mapping[int, tuple[int, ...]]] = {}
    flat_route_sets: dict[tuple[int, int], frozenset[int]] = {}
    flat_stand_route_sets: dict[tuple[int, int], frozenset[int]] = {}

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

        for cell, distance in distances.items():
            flat_distances[(target, cell)] = distance
        predecessor_maps[target] = MappingProxyType(predecessors)
        stand_maps[target] = MappingProxyType(stand_cells)

        target_routes = {}
        target_stand_routes = {}
        for cell in distances:
            r_tuple = _build_route_cells(cell, target, predecessors, stand_cells)
            s_tuple = _build_stand_route_cells(cell, target, predecessors, stand_cells)
            target_routes[cell] = r_tuple
            target_stand_routes[cell] = s_tuple
            flat_route_sets[(target, cell)] = frozenset(r_tuple)
            flat_stand_route_sets[(target, cell)] = frozenset(s_tuple)

        route_maps[target] = MappingProxyType(target_routes)
        stand_route_maps[target] = MappingProxyType(target_stand_routes)

    return (
        MappingProxyType(flat_distances),
        MappingProxyType(predecessor_maps),
        MappingProxyType(stand_maps),
        MappingProxyType(route_maps),
        MappingProxyType(stand_route_maps),
        MappingProxyType(flat_route_sets),
        MappingProxyType(flat_stand_route_sets),
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


def _build_stand_route_cells(
    cell: int,
    target: int,
    predecessors: Mapping[int, int],
    stand_cells: Mapping[int, int],
) -> tuple[int, ...]:
    if cell == target:
        return ()
    stands: list[int] = []
    node = cell
    while node != target:
        stand = stand_cells.get(node)
        if stand is None:
            break
        stands.append(stand)
        node = predecessors[node]
    return tuple(stands)


def _compute_dead_cells_for_o(
    board: StaticBoard,
    push_distances: Mapping[tuple[int, int], int],
) -> frozenset[int]:
    reachable_cells = {cell for (_target, cell) in push_distances}
    return frozenset(board.floor - reachable_cells)


def _compute_reachable_o_pairs(
    board: StaticBoard,
    push_distances: Mapping[tuple[int, int], int],
) -> frozenset[tuple[int, int]]:
    reachable_pairs: set[tuple[int, int]] = set()
    for first, second in itertools.combinations(sorted(board.floor), 2):
        for line in board.win_lines:
            for player_target in line:
                targets = tuple(cell for cell in line if cell != player_target)
                if len(targets) != 2:
                    continue
                if (
                    (targets[0], first) in push_distances
                    and (targets[1], second) in push_distances
                ) or (
                    (targets[1], first) in push_distances
                    and (targets[0], second) in push_distances
                ):
                    reachable_pairs.add((first, second))
                    break
            if (first, second) in reachable_pairs:
                break
    return frozenset(reachable_pairs)

