from __future__ import annotations

import heapq
import itertools
import os
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from solver.randomPythonFiles.superTicTacGoSolver import (
    apply_single_move,
    normalize_board,
)


DIRECTIONS = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)
MOVABLE_PIECES = {"X", "O"}
USEFUL_PIECES = {"O", "U"}
EMPTY = "."


@dataclass(frozen=True)
class Geometry:
    rows: int
    cols: int
    barriers: frozenset[int]
    lines: tuple[tuple[int, int, int], ...]
    valid_lines: tuple[tuple[int, int, int], ...]
    neighbors: tuple[tuple[tuple[str, int, int], ...], ...]


@dataclass(frozen=True)
class Parent:
    previous: str | None
    segment: str


@lru_cache(maxsize=None)
def _geometry(rows: int, cols: int, barriers: frozenset[int]) -> Geometry:
    lines = []
    for row in range(rows):
        base = row * cols
        for col in range(cols - 2):
            idx = base + col
            lines.append((idx, idx + 1, idx + 2))

    for row in range(rows - 2):
        for col in range(cols):
            idx = (row * cols) + col
            lines.append((idx, idx + cols, idx + (cols * 2)))

    valid_lines = tuple(line for line in lines if not any(idx in barriers for idx in line))

    neighbors = []
    for idx in range(rows * cols):
        row, col = divmod(idx, cols)
        cell_neighbors = []
        for move, row_delta, col_delta in DIRECTIONS:
            next_row = row + row_delta
            next_col = col + col_delta
            if 0 <= next_row < rows and 0 <= next_col < cols:
                next_idx = (next_row * cols) + next_col
                landing_row = row + (row_delta * 2)
                landing_col = col + (col_delta * 2)
                landing_idx = -1
                if 0 <= landing_row < rows and 0 <= landing_col < cols:
                    landing_idx = (landing_row * cols) + landing_col
                cell_neighbors.append((move, next_idx, landing_idx))
        neighbors.append(tuple(cell_neighbors))

    return Geometry(
        rows=rows,
        cols=cols,
        barriers=barriers,
        lines=tuple(lines),
        valid_lines=valid_lines,
        neighbors=tuple(neighbors),
    )


def _to_key(board: tuple[tuple[str, ...], ...]) -> str:
    return "".join(EMPTY if cell == "" else cell for row in board for cell in row)


def _geometry_for_board(board: tuple[tuple[str, ...], ...]) -> Geometry:
    rows = len(board)
    cols = len(board[0])
    barriers = frozenset(
        (row * cols) + col
        for row, cells in enumerate(board)
        for col, cell in enumerate(cells)
        if cell == "B"
    )
    return _geometry(rows, cols, barriers)


def _is_solved(key: str, geometry: Geometry) -> bool:
    return any(
        key[first] in USEFUL_PIECES
        and key[second] in USEFUL_PIECES
        and key[third] in USEFUL_PIECES
        for first, second, third in geometry.lines
    )


def _is_lost(key: str, geometry: Geometry) -> bool:
    return any(
        key[first] == "X" and key[second] == "X" and key[third] == "X"
        for first, second, third in geometry.lines
    )


def _piece_can_move(key: str, geometry: Geometry, idx: int) -> bool:
    for _, push_from, push_to in geometry.neighbors[idx]:
        if push_to == -1:
            continue
        if key[push_from] not in (EMPTY, "U"):
            continue
        if key[push_to] == EMPTY:
            return True
    return False


def _spot_can_become_user(key: str, geometry: Geometry, idx: int) -> bool:
    cell = key[idx]
    return cell in (EMPTY, "U") or (cell == "X" and _piece_can_move(key, geometry, idx))


def _spot_can_become_empty(key: str, geometry: Geometry, idx: int) -> bool:
    cell = key[idx]
    return cell == EMPTY or (cell == "X" and _piece_can_move(key, geometry, idx))


def _useful_piece_can_move(key: str, geometry: Geometry, idx: int) -> bool:
    for _, push_from, push_to in geometry.neighbors[idx]:
        if push_to == -1:
            continue
        if _spot_can_become_user(key, geometry, push_from) and _spot_can_become_empty(
            key, geometry, push_to
        ):
            return True
    return False


def _soft_locked(key: str, geometry: Geometry) -> bool:
    o_locations = [idx for idx, cell in enumerate(key) if cell == "O"]
    if len(o_locations) != 2:
        return False

    first, second = o_locations
    first_row, first_col = divmod(first, geometry.cols)
    second_row, second_col = divmod(second, geometry.cols)
    if first_row == second_row or first_col == second_col:
        return False

    return not any(_useful_piece_can_move(key, geometry, idx) for idx in o_locations)


def _is_dead(key: str, geometry: Geometry) -> bool:
    if sum(cell in USEFUL_PIECES for cell in key) < 3:
        return True
    return not geometry.valid_lines


def _pruned(key: str, geometry: Geometry) -> bool:
    return _is_lost(key, geometry) or _is_dead(key, geometry) or _soft_locked(key, geometry)


def _line_score(key: str, line: tuple[int, int, int], useful: tuple[int, ...], cols: int) -> int:
    occupied = 0
    blockers = 0
    distance = 0

    for target in line:
        cell = key[target]
        if cell in USEFUL_PIECES:
            occupied += 1
        elif cell == "X":
            blockers += 1

        target_row, target_col = divmod(target, cols)
        distance += min(
            abs(piece_row - target_row) + abs(piece_col - target_col)
            for piece_row, piece_col in (divmod(piece, cols) for piece in useful)
        )

    return distance - (occupied * 3) + (blockers * 4)


@lru_cache(maxsize=250_000)
def _best_line_score(key: str, geometry: Geometry) -> int:
    if _is_solved(key, geometry):
        return 0

    useful = tuple(idx for idx, cell in enumerate(key) if cell in USEFUL_PIECES)
    if len(useful) < 3:
        return 1_000_000

    return max(
        0,
        min(
            (
                _line_score(key, line, useful, geometry.cols)
                for line in geometry.valid_lines
            ),
            default=1_000_000,
        ),
    )


def _heuristic(key: str, geometry: Geometry) -> int:
    return _best_line_score(key, geometry)


def _reachable_paths(key: str, geometry: Geometry) -> dict[int, str]:
    start = key.index("U")
    queue = deque([start])
    paths = {start: ""}

    while queue:
        current = queue.popleft()
        current_path = paths[current]
        for move, next_idx, _ in geometry.neighbors[current]:
            if next_idx in paths or key[next_idx] != EMPTY:
                continue
            paths[next_idx] = current_path + move
            queue.append(next_idx)

    return paths


def _move_user(key: str, user_idx: int, target_idx: int) -> str:
    if user_idx == target_idx:
        return key

    cells = list(key)
    cells[user_idx] = EMPTY
    cells[target_idx] = "U"
    return "".join(cells)


def _push_from(key: str, user_idx: int, move: str, piece_idx: int, landing_idx: int) -> str | None:
    if landing_idx == -1:
        return None
    piece = key[piece_idx]
    if piece not in MOVABLE_PIECES or key[landing_idx] != EMPTY:
        return None

    cells = list(key)
    cells[user_idx] = EMPTY
    cells[piece_idx] = "U"
    cells[landing_idx] = piece
    return "".join(cells)


def _move_score(
    next_key: str,
    geometry: Geometry,
    segment: str,
    pushed_piece: str | None,
    current_score: int,
) -> tuple[int, int, int]:
    next_score = _best_line_score(next_key, geometry)
    improvement = current_score - next_score
    useful_push_bonus = 1 if pushed_piece == "O" else 0
    return (-improvement, -useful_push_bonus, len(segment))


def _next_states(key: str, geometry: Geometry):
    paths = _reachable_paths(key, geometry)
    user_idx = key.index("U")
    candidates = []
    current_score = _best_line_score(key, geometry)

    for target_idx, walk_path in paths.items():
        if not walk_path:
            continue
        walked_key = _move_user(key, user_idx, target_idx)
        if _is_solved(walked_key, geometry):
            candidates.append(
                (
                    _move_score(
                        walked_key,
                        geometry,
                        walk_path,
                        None,
                        current_score,
                    ),
                    walked_key,
                    walk_path,
                )
            )

    for target_idx, walk_path in paths.items():
        walked_key = _move_user(key, user_idx, target_idx)
        for move, piece_idx, landing_idx in geometry.neighbors[target_idx]:
            pushed_piece = walked_key[piece_idx]
            pushed_key = _push_from(walked_key, target_idx, move, piece_idx, landing_idx)
            if pushed_key is None:
                continue
            segment = walk_path + move
            candidates.append(
                (
                    _move_score(
                        pushed_key,
                        geometry,
                        segment,
                        pushed_piece,
                        current_score,
                    ),
                    pushed_key,
                    segment,
                )
            )

    candidates.sort(key=lambda candidate: candidate[0])
    for _, next_key, segment in candidates:
        yield next_key, segment


def _reconstruct(parents: dict[str, Parent], key: str) -> str:
    segments = []
    current = key
    while True:
        parent = parents[current]
        if parent.previous is None:
            break
        segments.append(parent.segment)
        current = parent.previous
    return "".join(reversed(segments))


def _replay(
    start_board: tuple[tuple[str, ...], ...],
    moves: str,
) -> tuple[tuple[str, ...], ...]:
    board = start_board
    for move in moves:
        board = apply_single_move(board, move)
    return board


def _validated_result(
    start_board: tuple[tuple[str, ...], ...],
    geometry: Geometry,
    moves: str,
) -> tuple[str, tuple[tuple[str, ...], ...]] | None:
    final_board = _replay(start_board, moves)
    final_key = _to_key(final_board)
    if _is_solved(final_key, geometry):
        return moves, final_board
    return None


def _weight_for_mode(mode: str) -> float:
    if mode == "fast":
        return 2.0
    if mode == "exact":
        return 0.0
    return 1.35


def solve(
    start_board,
    progress_every: int = 100_000,
    max_states: int | None = None,
    mode: str | None = None,
):
    start_board = normalize_board(start_board)
    geometry = _geometry_for_board(start_board)
    start_key = _to_key(start_board)
    mode = (mode or os.getenv("SOLVER_MODE") or "hybrid").strip().lower()
    if mode not in {"hybrid", "fast", "exact"}:
        mode = "hybrid"

    if _is_solved(start_key, geometry):
        return "", start_board, 1
    if _pruned(start_key, geometry):
        return None, None, 1

    queue = []
    counter = itertools.count()
    parents = {start_key: Parent(previous=None, segment="")}
    best_cost_seen = {start_key: 0}
    states_checked = 0
    best_solution: tuple[str, tuple[tuple[str, ...], ...], int] | None = None
    weight = _weight_for_mode(mode)

    heapq.heappush(
        queue,
        (_heuristic(start_key, geometry) * weight, 0, next(counter), start_key),
    )

    while queue:
        _priority, cost_so_far, _, current_key = heapq.heappop(queue)
        if cost_so_far != best_cost_seen.get(current_key):
            continue

        if best_solution and cost_so_far >= len(best_solution[0]):
            continue

        states_checked += 1
        if progress_every and states_checked % progress_every == 0:
            print(states_checked, flush=True)

        if _pruned(current_key, geometry):
            if max_states is not None and states_checked >= max_states:
                break
            continue

        if _is_solved(current_key, geometry):
            moves = _reconstruct(parents, current_key)
            validated = _validated_result(start_board, geometry, moves)
            if validated is not None:
                solution_moves, final_board = validated
                best_solution = (solution_moves, final_board, states_checked)
                if mode == "fast":
                    break
                if mode == "hybrid":
                    weight = 1.0
            if max_states is not None and states_checked >= max_states:
                break
            continue

        if max_states is not None and states_checked >= max_states:
            break

        for next_key, segment in _next_states(current_key, geometry):
            if _pruned(next_key, geometry):
                continue

            next_cost = cost_so_far + len(segment)
            if best_solution and next_cost >= len(best_solution[0]):
                continue
            if next_cost >= best_cost_seen.get(next_key, 1_000_000_000):
                continue

            best_cost_seen[next_key] = next_cost
            parents[next_key] = Parent(previous=current_key, segment=segment)
            next_priority = next_cost + (_heuristic(next_key, geometry) * weight)
            heapq.heappush(
                queue,
                (next_priority, next_cost, next(counter), next_key),
            )

    if best_solution is None:
        return None, None, states_checked
    return best_solution
