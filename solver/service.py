import time
from typing import Any

from solver.randomPythonFiles.superTicTacGoSolver import (
    apply_single_move,
    normalize_board,
    solve,
)


SOLVER_NAME = "bfs"


class SolverError(ValueError):
    """Raised when the API receives a board the solver cannot accept."""


def _to_wire_board(board: tuple[tuple[str, ...], ...]) -> list[list[str]]:
    return [list(row) for row in board]


def _build_steps(
    start_board: tuple[tuple[str, ...], ...],
    moves: str | None,
) -> list[dict[str, Any]]:
    if not moves:
        return []

    board = start_board
    steps = []
    for move in moves:
        board = apply_single_move(board, move)
        steps.append({"move": move, "board": _to_wire_board(board)})

    return steps


def solve_board(board: list[list[str]], max_states: int | None = None) -> dict[str, Any]:
    try:
        start_board = normalize_board(board)
    except ValueError as exc:
        raise SolverError(str(exc)) from exc

    start_time = time.perf_counter()
    moves, final_board, states_checked = solve(
        start_board,
        progress_every=0,
        max_states=max_states,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "solved": moves is not None,
        "moves": moves,
        "states_checked": states_checked,
        "elapsed_ms": elapsed_ms,
        "start_board": _to_wire_board(start_board),
        "final_board": _to_wire_board(final_board) if final_board else None,
        "steps": _build_steps(start_board, moves),
    }
