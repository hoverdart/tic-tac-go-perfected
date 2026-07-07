"""Push-path and concrete keystroke reconstruction."""

from __future__ import annotations

from collections import deque

from solver.push_solver.models import (
    DIRECTION_BY_MOVE,
    DIRECTIONS,
    GoalInfo,
    Parent,
    Push,
    State,
    StaticBoard,
)


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

