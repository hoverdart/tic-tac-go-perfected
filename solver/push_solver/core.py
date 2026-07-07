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
from dataclasses import dataclass, replace
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
    adjacency: tuple[tuple[int, ...], ...]
    push_transitions: tuple[tuple[tuple[str, int, int], ...], ...]
    push_route_sets: Mapping[int, Mapping[int, frozenset[int]]]
    reachable_o_pairs: frozenset[tuple[int, int]]

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
    push: Push | None = None
    pushes: tuple[Push, ...] = ()

    def push_tuple(self) -> tuple[Push, ...]:
        if self.pushes:
            return self.pushes
        if self.push is not None:
            return (self.push,)
        return ()


@dataclass(frozen=True)
class GoalInfo:
    line: tuple[int, int, int]
    player_target: int


@dataclass(frozen=True)
class LinePlan:
    score: float
    line: tuple[int, int, int]
    player_target: int
    o_targets: tuple[int, int]
    o_assignment: tuple[tuple[int, int], tuple[int, int]]
    route_cells: frozenset[int]
    stand_cells: frozenset[int]
    player_target_has_x: bool
    player_target_reachable: bool
    x_on_line_count: int
    blocked_route_count: int


@dataclass(frozen=True)
class SearchAttempt:
    strategy: str
    solved: bool
    nodes_expanded: int
    peak_closed_size: int
    elapsed_ms: float
    failure_reason: str | None


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
    strategy: str | None = None
    attempts: tuple[SearchAttempt, ...] = ()


@dataclass(frozen=True)
class SearchStrategyConfig:
    name: str
    kind: str = "weighted"
    weight: float = 2.0
    g_weight: float = 1.0
    bias_scale: float = 1.0
    use_macros: bool = False
    policy_weight: float = 0.0


@dataclass
class SearchContext:
    static: StaticBoard
    start_state: State
    initial_player: int
    target_access_penalty: float
    region_cache: dict[State, frozenset[int]]
    h_cache: dict[State, float]
    plan_cache: dict[State, LinePlan | None]
    top_plan_cache: dict[State, tuple[LinePlan, ...]]
    o_push_count_cache: dict[State, int]
    successor_cache: dict[State, tuple[tuple[Push, State, frozenset[int], float, float], ...]]
    policy_score_cache: dict[tuple[State, tuple[Push, ...], State], float]


StrategyChild = tuple[tuple[Push, ...], State, frozenset[int], float, float, int, float]


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
    
    # Precompute structural invariants for fast inner-loop generation
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
        adjacency=tuple(adjacency_list),
        push_transitions=tuple(transitions_list),
        push_route_sets=MappingProxyType({}),
        reachable_o_pairs=frozenset(),
    )
    
    (
        push_distances,
        push_predecessors,
        push_stand_cells,
        push_routes,
        push_route_sets,
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
        adjacency=static_temp.adjacency,
        push_transitions=static_temp.push_transitions,
        push_route_sets=push_route_sets,
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


def is_x_loss(xs: frozenset[int], board: StaticBoard) -> bool:
    return any(all(cell in xs for cell in line) for line in board.win_lines)


def _compute_push_distance_maps(
    board: StaticBoard,
) -> tuple[
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, int]],
    Mapping[int, Mapping[int, tuple[int, ...]]],
    Mapping[int, Mapping[int, frozenset[int]]],
]:
    targets = {cell for line in board.win_lines for cell in line}
    distance_maps: dict[int, Mapping[int, int]] = {}
    predecessor_maps: dict[int, Mapping[int, int]] = {}
    stand_maps: dict[int, Mapping[int, int]] = {}
    route_maps: dict[int, Mapping[int, tuple[int, ...]]] = {}
    route_sets_maps: dict[int, Mapping[int, frozenset[int]]] = {}
    
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
        
        target_routes = {}
        target_route_sets = {}
        for cell in distances:
            r_tuple = _build_route_cells(cell, target, predecessors, stand_cells)
            target_routes[cell] = r_tuple
            target_route_sets[cell] = frozenset(r_tuple)
            
        route_maps[target] = MappingProxyType(target_routes)
        route_sets_maps[target] = MappingProxyType(target_route_sets)

    return (
        MappingProxyType(distance_maps),
        MappingProxyType(predecessor_maps),
        MappingProxyType(stand_maps),
        MappingProxyType(route_maps),
        MappingProxyType(route_sets_maps),
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


def _compute_reachable_o_pairs(
    board: StaticBoard,
    push_distances: Mapping[int, Mapping[int, int]],
) -> frozenset[tuple[int, int]]:
    reachable_pairs: set[tuple[int, int]] = set()
    for first, second in itertools.combinations(sorted(board.floor), 2):
        for line in board.win_lines:
            for player_target in line:
                targets = tuple(cell for cell in line if cell != player_target)
                if len(targets) != 2:
                    continue
                if (
                    first in push_distances.get(targets[0], {})
                    and second in push_distances.get(targets[1], {})
                ) or (
                    first in push_distances.get(targets[1], {})
                    and second in push_distances.get(targets[0], {})
                ):
                    reachable_pairs.add((first, second))
                    break
            if (first, second) in reachable_pairs:
                break
    return frozenset(reachable_pairs)


def _direction_permanently_blocked(
    cell: int,
    move: str,
    board: StaticBoard,
    permanent_blockers: frozenset[int],
) -> bool:
    row, col = board.coord(cell)
    dr, dc = DIRECTION_BY_MOVE[move]
    dest_row = row + dr
    dest_col = col + dc
    stand_row = row - dr
    stand_col = col - dc
    if not board.in_bounds(dest_row, dest_col):
        return True
    if not board.in_bounds(stand_row, stand_col):
        return True
    dest = board.index(dest_row, dest_col)
    stand = board.index(stand_row, stand_col)
    return dest in permanent_blockers or stand in permanent_blockers


def _piece_permanently_frozen(
    cell: int,
    board: StaticBoard,
    permanent_blockers: frozenset[int],
) -> bool:
    horizontal = _direction_permanently_blocked(
        cell,
        "L",
        board,
        permanent_blockers,
    ) and _direction_permanently_blocked(cell, "R", board, permanent_blockers)
    vertical = _direction_permanently_blocked(
        cell,
        "U",
        board,
        permanent_blockers,
    ) and _direction_permanently_blocked(cell, "D", board, permanent_blockers)
    return horizontal and vertical


def frozen_pieces(
    os: frozenset[int],
    xs: frozenset[int],
    board: StaticBoard,
) -> tuple[frozenset[int], frozenset[int]]:
    """Return pieces proven immovable by walls/edges/proven-frozen pieces.

    This intentionally under-approximates Sokoban freeze detection. It only
    adds a movable piece to the permanent blocker set after that piece is
    already frozen using the current permanent set.
    """
    pieces = os | xs
    frozen: set[int] = set()
    changed = True
    while changed:
        changed = False
        permanent_blockers = frozenset(board.walls | frozen)
        for cell in sorted(pieces - frozen):
            if _piece_permanently_frozen(cell, board, permanent_blockers):
                frozen.add(cell)
                changed = True
    frozen_set = frozenset(frozen)
    return frozen_set & os, frozen_set & xs


def _cell_is_on_win_line(cell: int, board: StaticBoard) -> bool:
    return any(cell in line for line in board.win_lines)


def _floor_reachable_with_permanent_blockers(
    start: int,
    target: int,
    board: StaticBoard,
    permanent_blockers: frozenset[int],
) -> bool:
    if start == target:
        return True
    if start in permanent_blockers or target in permanent_blockers:
        return False
    queue = [start]
    seen = {start}
    for current in queue:
        for nxt in board.adjacency[current]:
            if nxt in permanent_blockers or nxt in seen:
                continue
            if nxt == target:
                return True
            seen.add(nxt)
            queue.append(nxt)
    return False


def _push_reachable_with_permanent_blockers(
    start: int,
    target: int,
    board: StaticBoard,
    permanent_blockers: frozenset[int],
) -> bool:
    if start == target:
        return True
    if start in permanent_blockers or target in permanent_blockers:
        return False
    distances = {target}
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
            if previous in permanent_blockers or stand in permanent_blockers:
                continue
            if previous == start:
                return True
            if previous not in distances:
                distances.add(previous)
                queue.append(previous)
    return False


def _assignment_survives_frozen_constraints(
    *,
    o_cells: tuple[int, ...],
    targets: tuple[int, int],
    frozen_os: frozenset[int],
    frozen_xs: frozenset[int],
    board: StaticBoard,
) -> bool:
    permanent_blockers = frozenset(board.walls | frozen_xs | frozen_os)
    for assigned in (
        ((o_cells[0], targets[0]), (o_cells[1], targets[1])),
        ((o_cells[0], targets[1]), (o_cells[1], targets[0])),
    ):
        valid = True
        for o_cell, target in assigned:
            if o_cell in frozen_os:
                if target != o_cell:
                    valid = False
                    break
                continue
            if target in permanent_blockers:
                valid = False
                break
            if not _push_reachable_with_permanent_blockers(
                o_cell,
                target,
                board,
                permanent_blockers,
            ):
                valid = False
                break
        if valid:
            return True
    return False


def _has_viable_line_under_frozen_constraints(
    os: frozenset[int],
    frozen_os: frozenset[int],
    frozen_xs: frozenset[int],
    board: StaticBoard,
    *,
    player: int | None,
) -> bool:
    if len(os) != 2:
        return True
    o_cells = tuple(os)
    permanent_blockers = frozenset(board.walls | frozen_xs | frozen_os)
    for line in board.win_lines:
        if frozen_xs & set(line):
            continue
        if not frozen_os.issubset(line):
            continue
        for player_target in line:
            if player_target in permanent_blockers:
                continue
            if player is not None and not _floor_reachable_with_permanent_blockers(
                player,
                player_target,
                board,
                permanent_blockers,
            ):
                continue
            targets = tuple(cell for cell in line if cell != player_target)
            if len(targets) != 2:
                continue
            if _assignment_survives_frozen_constraints(
                o_cells=o_cells,
                targets=targets,
                frozen_os=frozen_os,
                frozen_xs=frozen_xs,
                board=board,
            ):
                return True
    return False


def is_deadlock(
    os: frozenset[int],
    xs: frozenset[int],
    board: StaticBoard,
    *,
    player: int | None = None,
) -> bool:
    if any(cell in board.dead_cells_for_o for cell in os):
        return True
    if len(os) == 2:
        pair = tuple(sorted(os))
        if pair not in board.reachable_o_pairs:
            return True

    frozen_os, frozen_xs = frozen_pieces(os, xs, board)
    if not frozen_os and not frozen_xs:
        return False
    if any(not _cell_is_on_win_line(cell, board) for cell in frozen_os):
        return True
    return not _has_viable_line_under_frozen_constraints(
        os,
        frozen_os,
        frozen_xs,
        board,
        player=player,
    )


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
REACH_GAIN_BIAS: float = 1.80 
TARGET_ACCESS_UNBLOCK_BIAS: float = 9.0 
O_PUSH_ACCESS_GAIN_BIAS: float = 0.80
HEURISTIC_PROGRESS_BIAS: float = 1.00
PLAN_X_CLEAR_BIAS: float = 2.00
PLAN_TARGET_REACHABLE_BIAS: float = 2.50
PLAN_TARGET_UNREACHABLE_BIAS: float = 2.00


def _route_cells(cell: int, target: int, board: StaticBoard) -> tuple[int, ...]:
    """Box-landing cells and player-stand cells along the canonical shortest
    wall-only push route from `cell` to `target` (both endpoints included)."""
    return board.push_routes.get(target, {}).get(cell, ())


def _stand_cells(cell: int, target: int, board: StaticBoard) -> tuple[int, ...]:
    stand_cells = board.push_stand_cells.get(target, {})
    predecessors = board.push_predecessors.get(target, {})
    if cell == target or cell not in board.push_distances.get(target, {}):
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


def _occupancy_penalty(
    cell: int,
    target: int,
    blockers: frozenset[int],
    board: StaticBoard,
) -> float:
    base = board.push_distances.get(target, {}).get(cell, math.inf)
    if base == math.inf:
        return math.inf
    route_set = board.push_route_sets.get(target, {}).get(cell, frozenset())
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
    blockers0 = state.xs | {o_cells[1]}
    blockers1 = state.xs | {o_cells[0]}
    plans: list[LinePlan] = []
    seen: set[
        tuple[
            tuple[int, int, int],
            int,
            tuple[int, int],
            tuple[tuple[int, int], tuple[int, int]],
        ]
    ] = set()
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
            x_on_line_count = sum(1 for cell in line if cell in state.xs)
            plan_penalty += X_ON_LINE_PENALTY * x_on_line_count

            assignments = (
                (o_cells[0], targets[0], blockers0, o_cells[1], targets[1], blockers1),
                (o_cells[0], targets[1], blockers0, o_cells[1], targets[0], blockers1),
            )
            for o_a, target_a, blockers_a, o_b, target_b, blockers_b in assignments:
                cost_a = _occupancy_penalty(o_a, target_a, blockers_a, board)
                cost_b = _occupancy_penalty(o_b, target_b, blockers_b, board)
                if not math.isfinite(cost_a + cost_b):
                    continue

                o_assignment = ((o_a, target_a), (o_b, target_b))
                key = (line, player_target, targets, o_assignment)
                if key in seen:
                    continue
                seen.add(key)

                route_a = _route_cells(o_a, target_a, board)
                route_b = _route_cells(o_b, target_b, board)
                route_cells = frozenset(route_a + route_b)
                stand_cells = frozenset(
                    _stand_cells(o_a, target_a, board)
                    + _stand_cells(o_b, target_b, board)
                )
                blocked_route_count = len(state.xs & route_cells)
                score = cost_a + cost_b + plan_penalty

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

    plans.sort(key=lambda plan: plan.score)
    return tuple(plans[:limit])


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
    if region is None:
        region = reachable(state.player, state.os, state.xs, board)
    occupied = state.os | state.xs
    parent_top_plans = _top_line_plans_for(
        state,
        board,
        region=region,
        target_access_penalty=target_access_penalty,
        top_plan_cache=top_plan_cache,
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

                if is_deadlock(new_os, new_xs, board, player=cell):
                    continue
                
                next_state, next_region = _normalize_with_region(cell, new_os, new_xs, board)
                
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


def _reconstruct_pushes(
    parents: dict[State, Parent],
    goal_state: State,
) -> tuple[Push, ...]:
    chunks: list[tuple[Push, ...]] = []
    current = goal_state
    while current in parents:
        parent = parents[current]
        chunks.append(parent.push_tuple())
        current = parent.previous
    return tuple(
        push
        for chunk in reversed(chunks)
        for push in chunk
    )


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
    pushes: tuple[Push, ...],
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
        successor_cache={},
        policy_score_cache={},
    )


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
            )
        )
        context.successor_cache[state] = items
        for _push, nxt, child_region, child_h, _bias in items:
            context.region_cache.setdefault(nxt, child_region)
            context.h_cache.setdefault(nxt, child_h)
    return context.successor_cache[state]


def _same_piece_continuation(
    context: SearchContext,
    state: State,
    *,
    piece: str,
    cell: int,
    move: str,
) -> tuple[Push, State, frozenset[int], float, float] | None:
    piece_pushes = [
        item
        for item in _successors_for(context, state)
        if item[0].piece == piece and item[0].cell == cell
    ]
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
        # Macros reduce queue granularity, but their real cost remains push count.
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
) -> float:
    key = (parent, pushes, child)
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
    score = policy.score(features)
    context.policy_score_cache[key] = score
    return score


def _strategy_children_for(
    context: SearchContext,
    state: State,
    *,
    use_macros: bool,
    bias_scale: float,
    policy_weight: float,
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
                ),
            )
            for pushes, nxt, child_region, child_h, bias, push_cost, _policy_score in children
        ]
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


def _portfolio_configs(weight: float) -> tuple[tuple[SearchStrategyConfig, float], ...]:
    return (
        (
            SearchStrategyConfig(
                name="v1_weighted",
                kind="weighted",
                weight=weight,
                g_weight=1.0,
                bias_scale=1.0,
            ),
            0.30,
        ),
        (
            SearchStrategyConfig(
                name="greedy_low_g",
                kind="weighted",
                weight=max(weight, 2.2),
                g_weight=0.25,
                bias_scale=1.0,
            ),
            0.15,
        ),
        (
            SearchStrategyConfig(
                name="greedy_bias",
                kind="weighted",
                weight=max(weight, 2.5),
                g_weight=0.15,
                bias_scale=1.75,
            ),
            0.15,
        ),
        (
            SearchStrategyConfig(
                name="macro_greedy",
                kind="weighted",
                weight=max(weight, 2.5),
                g_weight=0.20,
                bias_scale=1.50,
                use_macros=True,
            ),
            0.20,
        ),
        (
            SearchStrategyConfig(
                name="rank_discrepancy",
                kind="rank_discrepancy",
                weight=weight,
                g_weight=1.0,
                bias_scale=1.25,
            ),
            0.45,
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
            1.00,
        ),
    )


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
    configs = _portfolio_configs(weight)
    context = _build_search_context(static, start_state, initial_player)

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
