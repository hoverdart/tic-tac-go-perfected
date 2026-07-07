"""Conservative deadlock checks for push-level Tic Tac Go states."""

from __future__ import annotations

from collections import deque

from solver.push_solver.models import DIRECTION_BY_MOVE, DIRECTIONS, StaticBoard


def is_x_loss(xs: frozenset[int], board: StaticBoard) -> bool:
    return any(all(cell in xs for cell in line) for line in board.win_lines)


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
    """Return pieces proven immovable by walls/edges/proven-frozen pieces."""
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

