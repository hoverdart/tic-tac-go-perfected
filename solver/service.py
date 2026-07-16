"""
Public interface between the FastAPI layer and the underlying solver implementations.

This module is the single entry point for solving a board. It handles:
  - Solver selection: routes larger boards to heuristic-CNN, otherwise reads
    SOLVER_IMPL and SOLVER_MODE from environment variables.
  - Daily solver portfolio: runs the push solver first, then the existing
    heuristic beam/CNN solver only when push search does not find a solution.
  - Fallback logic: if the optimized A* solver gives up (returns None) and budget
    remains, the legacy BFS solver gets a second attempt as a safety net.
  - Result normalization: regardless of which solver ran, solve_board() always
    returns the same dict shape so callers never need to know which path was taken.
"""

from __future__ import annotations

import os
import time
from typing import Any

from solver.board_utils import board_dimensions, normalize_board, to_wire_board
from solver import optimized_solver
from solver.legacy_solver import (
    apply_single_move,
    solve as legacy_solve,
)


DAILY_PUSH_MAX_NODES = 500_000
DAILY_PUSH_TIMEOUT_SECONDS = 30.0


def _solver_impl(board: tuple[tuple[str, ...], ...] | None = None) -> str:
    """Return the solver implementation to use, read from SOLVER_IMPL env var.

    Valid values: "legacy" (BFS), "optimized" (A*), "learned" (linear
    child-path ranked A*), or "push" (classical push-level A*). Defaults to
    "legacy". SOLVER_IMPL=push is explicitly opt-in for all board sizes;
    otherwise larger boards route to heuristic-CNN.
    """

    rows, cols = board_dimensions(board) if board is not None else (0, 0)
    requested_impl = os.getenv("SOLVER_IMPL", "legacy").strip().lower()
    if requested_impl == "push":
        impl = "push"
    elif rows >= 6 and cols >= 6:
        impl = "heuristiccnn"
    else:
        impl = requested_impl
    if impl not in {"legacy", "optimized", "learned", "push", "heuristiccnn"}:
        return "legacy"
    return impl


def _solver_mode(board: tuple[tuple[str, ...], ...] | None = None) -> str:
    """Return the search mode for the optimized solver, read from SOLVER_MODE.

    Valid values:
      "fast"   — greedy weighted A* (weight=2.0), returns the first solution found.
      "hybrid" — starts greedy (weight=1.35), then relaxes to exact (weight=1.0)
                 after the first solution to search for a shorter one.
      "exact"  — unweighted Dijkstra (weight=0.0), guarantees the optimal solution.

    Like _solver_impl(), this is evaluated once at import time.
    """
    rows, cols = board_dimensions(board) if board is not None else (0, 0)
    if rows >= 6 and cols >= 6:
        mode = "N/A"
    else:
        mode = os.getenv("SOLVER_MODE", "hybrid").strip().lower()
    if mode not in {"hybrid", "fast", "exact", "N/A"}:
        return "hybrid"
    return mode


def _solver_name(
    impl: str,
    board: tuple[tuple[str, ...], ...],
    used_fallback: bool = False,
) -> str:
    if impl == "heuristiccnn":
        name = "heuristic-CNN"
    elif impl == "push":
        name = "push-v2"
    elif impl == "learned":
        name = f"linear-tree-v1-{_solver_mode(board)}"
    elif impl == "optimized":
        name = f"optimized-{_solver_mode(board)}"
    else:
        name = "bfs"

    if used_fallback:
        name = f"{name}+bfs-fallback"
    return name


class SolverError(ValueError):
    """Raised when the API receives a board the solver cannot accept."""


def _build_steps(
    start_board: tuple[tuple[str, ...], ...],
    moves: str | None,
) -> list[dict[str, Any]]:
    """Re-simulate each move from the start board and record the resulting board state.

    The solver returns only the final board, not the intermediate boards for each
    step. To produce the per-move breakdown the API exposes, we replay the move
    string one character at a time using apply_single_move.

    Returns an empty list if moves is None or empty (unsolved or already solved).
    """
    if not moves:
        return []

    board = start_board
    steps = []
    for move in moves:
        board = apply_single_move(board, move)
        steps.append({"move": move, "board": to_wire_board(board)})

    return steps


def _build_result(
    start_board: tuple[tuple[str, ...], ...],
    *,
    moves: str | None,
    final_board: tuple[tuple[str, ...], ...] | None,
    states_checked: int,
    solver_name: str,
    start_time: float,
) -> dict[str, Any]:
    """Normalize a solver attempt into the API response shape."""
    return {
        "solved": moves is not None,
        "solver_name": solver_name,
        "moves": moves,
        "states_checked": states_checked,
        "elapsed_ms": (time.perf_counter() - start_time) * 1000,
        "start_board": to_wire_board(start_board),
        "final_board": to_wire_board(final_board) if final_board else None,
        "steps": _build_steps(start_board, moves),
    }


def _run_push_solver(
    board: tuple[tuple[str, ...], ...],
    *,
    max_nodes: int,
    timeout_seconds: float,
) -> Any:
    from solver.push_solver import solve as push_solve

    return push_solve(
        board,
        max_nodes=max_nodes,
        timeout_seconds=timeout_seconds,
    )


def _run_heuristic_cnn_solver(
    board: tuple[tuple[str, ...], ...],
) -> tuple[str | None, tuple[tuple[str, ...], ...] | None, int]:
    from solver import heuristic_cnn_solver

    return heuristic_cnn_solver.solve(board)


def solve_daily_board(board: list[list[str]]) -> dict[str, Any]:
    """Run the production daily portfolio: push first, then Beam/CNN fallback."""
    try:
        start_board = normalize_board(board)
    except ValueError as exc:
        raise SolverError(str(exc)) from exc

    start_time = time.perf_counter()
    push_result = _run_push_solver(
        start_board,
        max_nodes=DAILY_PUSH_MAX_NODES,
        timeout_seconds=DAILY_PUSH_TIMEOUT_SECONDS,
    )
    moves = push_result.moves
    final_board = push_result.final_board
    states_checked = push_result.nodes_expanded
    solver_name = "push-v2"

    if moves is None:
        fallback_moves, _fallback_board, fallback_states = _run_heuristic_cnn_solver(
            start_board
        )
        states_checked += fallback_states
        solver_name = "push-v2+heuristic-CNN-fallback"

        if fallback_moves is not None:
            from solver.push_solver import verify_solution

            verification = verify_solution(start_board, fallback_moves)
            if not verification.ok:
                raise RuntimeError(
                    "Heuristic-CNN fallback returned an invalid solution: "
                    f"{verification.error}"
                )
            moves = fallback_moves
            final_board = verification.final_board

    return _build_result(
        start_board,
        moves=moves,
        final_board=final_board,
        states_checked=states_checked,
        solver_name=solver_name,
        start_time=start_time,
    )


def solve_board(board: list[list[str]], max_states: int | None = None) -> dict[str, Any]:
    """Solve a Tic Tac Go board and return a normalized result dict.

    Routing logic:
      - If SOLVER_IMPL=optimized, runs the A* solver first.
      - If SOLVER_IMPL=learned, runs the Linear Tree Solver first.
      - If SOLVER_IMPL=push, runs the classical push-level solver first.
      - If the A* solver returns None (no solution found within budget) AND there
        are still states left in the budget AND SOLVER_FALLBACK=legacy, the legacy
        BFS solver gets to try with whatever budget remains. This acts as a safety
        net for edge cases the optimized solver's pruning might incorrectly discard.
      - If SOLVER_IMPL=legacy (or unset), runs the BFS solver directly.

    The `remaining_states` / `can_fallback` check ensures we never invoke the
    fallback solver when the optimized solver already exhausted the full budget —
    there would be nothing left for it to try.

    Args:
        board: 2-D list of strings representing the board (e.g. from JSON body).
        max_states: optional cap on total states explored across both solvers.

    Returns:
        A dict with keys: solved, moves, states_checked, elapsed_ms,
        start_board, final_board, steps.

    Raises:
        SolverError: if the board fails validation (wrong shape, missing U, etc.).
    """
    try:
        start_board = normalize_board(board)
    except ValueError as exc:
        raise SolverError(str(exc)) from exc

    start_time = time.perf_counter()
    solver_impl = _solver_impl(start_board)
    used_fallback = False
    if solver_impl == "heuristiccnn":
        from solver import heuristic_cnn_solver

        moves, final_board, states_checked = heuristic_cnn_solver.solve(start_board)
    elif solver_impl in {"optimized", "learned", "push"}:
        if solver_impl == "push":
            from solver.push_solver import solve as push_solve

            push_result = push_solve(
                start_board,
                max_nodes=500_000 if max_states is None else max_states,
            )
            moves = push_result.moves
            final_board = push_result.final_board
            states_checked = push_result.nodes_expanded
        elif solver_impl == "learned":
            from solver.learned_search import solver as learned_solver

            solve_fn = learned_solver.solve
            moves, final_board, states_checked = solve_fn(
                start_board,
                progress_every=0,
                max_states=max_states,
            )
        else:
            moves, final_board, states_checked = optimized_solver.solve(
                start_board,
                progress_every=0,
                max_states=max_states,
            )
        # Only fall back to legacy if budget wasn't fully consumed by the A* run.
        remaining_states = None if max_states is None else max_states - states_checked
        can_fallback = remaining_states is None or remaining_states > 0
        if (
            moves is None
            and solver_impl != "push"
            and can_fallback
            and os.getenv("SOLVER_FALLBACK", "legacy").strip().lower() == "legacy"
        ):
            fallback_moves, fallback_board, fallback_states = legacy_solve(
                start_board,
                progress_every=0,
                max_states=remaining_states,
            )
            used_fallback = True
            states_checked += fallback_states
            moves, final_board = fallback_moves, fallback_board
    else:
        moves, final_board, states_checked = legacy_solve(
            start_board,
            progress_every=0,
            max_states=max_states,
        )
    return _build_result(
        start_board,
        moves=moves,
        final_board=final_board,
        states_checked=states_checked,
        solver_name=_solver_name(solver_impl, start_board, used_fallback),
        start_time=start_time,
    )
