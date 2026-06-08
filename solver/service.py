from __future__ import annotations

import os
import time
from typing import Any

from solver import optimized_solver
from solver.randomPythonFiles.superTicTacGoSolver import (
    apply_single_move,
    solve as legacy_solve,
    normalize_board,
)


def _solver_impl() -> str:
    impl = os.getenv("SOLVER_IMPL", "legacy").strip().lower()
    if impl not in {"legacy", "optimized"}:
        return "legacy"
    return impl


def _solver_mode() -> str:
    mode = os.getenv("SOLVER_MODE", "hybrid").strip().lower()
    if mode not in {"hybrid", "fast", "exact"}:
        return "hybrid"
    return mode


SOLVER_NAME = f"optimized-{_solver_mode()}" if _solver_impl() == "optimized" else "bfs"


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
    solver_impl = _solver_impl()
    if solver_impl == "optimized":
        moves, final_board, states_checked = optimized_solver.solve(
            start_board,
            progress_every=0,
            max_states=max_states,
        )
        remaining_states = None if max_states is None else max_states - states_checked
        can_fallback = remaining_states is None or remaining_states > 0
        if (
            moves is None
            and can_fallback
            and os.getenv("SOLVER_FALLBACK", "legacy").strip().lower() == "legacy"
        ):
            fallback_moves, fallback_board, fallback_states = legacy_solve(
                start_board,
                progress_every=0,
                max_states=remaining_states,
            )
            states_checked += fallback_states
            moves, final_board = fallback_moves, fallback_board
    else:
        moves, final_board, states_checked = legacy_solve(
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
