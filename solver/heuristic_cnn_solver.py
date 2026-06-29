"""Heuristic + CNN beam-search solver wrapper.

This module intentionally stays thin: the actual search lives in
``solver.beam_search`` and the policy network lives in ``solver.small_cnn``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch as th

from solver.beam_search import beamSearch
from solver.board_utils import normalize_board
from solver.legacy_solver import (
    apply_single_move,
)
from solver.small_cnn import SmallCNN


MODEL_PATH = Path(__file__).with_name("small_cnn_policy.pt")

# Match the latest board-test scripts.
BEAM_WIDTH = 5000
BEAM_MAX_DEPTH = 200
BEAM_RESTARTS = 5
RANDOM_TIEBREAK = True
TIEBREAK_NOISE = 0.05
RANDOM_PREFIX_STEPS = [0, 0, 0, 5, 10]
CNN_MODEL_ACTION_WEIGHT = 1.0
RESTART_MODEL_ACTION_WEIGHTS = [0.1, 0.5, 1.0]
ATTEMPT_TIMEOUT_SECONDS = 300


_MODEL: SmallCNN | None = None


def load_model(model_path: str | Path = MODEL_PATH) -> SmallCNN:
    """Load the trained small CNN once and reuse it across solves."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = SmallCNN()
    state_dict = th.load(Path(model_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL = model
    return model


def replay_moves(
    start_board: tuple[tuple[str, ...], ...],
    moves: str,
) -> tuple[tuple[str, ...], ...]:
    """Apply a move string to get the final board."""
    board = start_board
    for move in moves:
        board = apply_single_move(board, move)
    return board


def _run_beam(
    start_board: tuple[tuple[str, ...], ...],
    model: SmallCNN | None,
    *,
    beam_width: int,
    max_depth: int,
    debug: bool,
    seed: int | None,
    model_action_weight: float,
    restart_model_action_weights: list[float] | None,
    timeout_seconds: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    return beamSearch(
        start_board,
        model,
        beam_width=beam_width,
        max_depth=max_depth,
        debug=debug,
        random_tiebreak=RANDOM_TIEBREAK,
        seed=seed,
        tiebreak_noise=TIEBREAK_NOISE,
        restarts=BEAM_RESTARTS,
        random_prefix_steps=RANDOM_PREFIX_STEPS,
        model_action_weight=model_action_weight,
        restart_model_action_weights=restart_model_action_weights,
        timeout_seconds=timeout_seconds,
    )


def solve(
    board: list[list[str]] | tuple[tuple[str, ...], ...],
    *,
    beam_width: int = BEAM_WIDTH,
    max_depth: int = BEAM_MAX_DEPTH,
    debug: bool = False,
    seed: int | None = None,
    model_path: str | Path = MODEL_PATH,
    attempt_timeout_seconds: int | None = ATTEMPT_TIMEOUT_SECONDS,
) -> tuple[str | None, tuple[tuple[str, ...], ...] | None, int]:
    """Solve a board with heuristic beam search, then CNN guidance if needed.

    Returns ``(moves, final_board, states_checked)`` to match the shape used by
    the other service solvers. ``beamSearch`` does not currently expose its true
    state-count value, so the third value is the replay transition count.
    """
    start_board = normalize_board(board)
    moves, transition_data = _run_beam(
        start_board,
        None,
        beam_width=beam_width,
        max_depth=max_depth,
        debug=debug,
        seed=seed,
        model_action_weight=0.0,
        restart_model_action_weights=None,
        timeout_seconds=attempt_timeout_seconds,
    )

    total_transition_count = len(transition_data)
    if not moves:
        model = load_model(model_path)
        moves, transition_data = _run_beam(
            start_board,
            model,
            beam_width=beam_width,
            max_depth=max_depth,
            debug=debug,
            seed=seed,
            model_action_weight=CNN_MODEL_ACTION_WEIGHT,
            restart_model_action_weights=RESTART_MODEL_ACTION_WEIGHTS,
            timeout_seconds=attempt_timeout_seconds,
        )
        total_transition_count += len(transition_data)

    if not moves:
        return None, None, total_transition_count

    final_board = replay_moves(start_board, moves)
    return moves, final_board, total_transition_count


def solve_board(
    board: list[list[str]] | tuple[tuple[str, ...], ...],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience dict wrapper for manual tests."""
    start_board = normalize_board(board)
    moves, final_board, states_checked = solve(start_board, **kwargs)
    return {
        "solved": moves is not None,
        "moves": moves,
        "states_checked": states_checked,
        "start_board": start_board,
        "final_board": final_board,
    }
