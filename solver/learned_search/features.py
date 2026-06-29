"""Feature extraction for learned child-path ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from solver import optimized_solver
from solver.legacy_solver import apply_single_move


@dataclass(frozen=True)
class CandidateFeatures:
    """Numeric description of one legal child path from a parent state."""

    parent_heuristic: float
    child_heuristic: float
    heuristic_delta: float
    segment_length: int
    pushed_o: int
    pushed_x: int
    walk_only: int
    child_solved: int
    child_lost: int
    child_pruned: int
    useful_line_occupancy: int
    x_threat_lines: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def key_to_board(
    key: str,
    rows: int,
    cols: int,
) -> tuple[tuple[str, ...], ...]:
    """Convert an optimized-solver key string back to a tuple board."""
    board = []
    for row in range(rows):
        start = row * cols
        cells = key[start : start + cols]
        board.append(tuple("" if cell == optimized_solver.EMPTY else cell for cell in cells))
    return tuple(board)


def segment_result_key(
    parent_key: str,
    geometry: optimized_solver.Geometry,
    segment: str,
) -> str:
    """Replay a compressed child segment from a key and return the resulting key."""
    board = key_to_board(parent_key, geometry.rows, geometry.cols)
    for move in segment:
        board = apply_single_move(board, move)
    return optimized_solver._to_key(board)


def _pushed_piece(parent_key: str, child_key: str) -> str | None:
    parent_o_positions = {
        index for index, cell in enumerate(parent_key) if cell == "O"
    }
    child_o_positions = {
        index for index, cell in enumerate(child_key) if cell == "O"
    }
    if parent_o_positions != child_o_positions:
        return "O"

    parent_x_positions = {
        index for index, cell in enumerate(parent_key) if cell == "X"
    }
    child_x_positions = {
        index for index, cell in enumerate(child_key) if cell == "X"
    }
    if parent_x_positions != child_x_positions:
        return "X"

    return None


def _useful_line_occupancy(key: str, geometry: optimized_solver.Geometry) -> int:
    if not geometry.valid_lines:
        return 0
    return max(
        sum(key[index] in optimized_solver.USEFUL_PIECES for index in line)
        for line in geometry.valid_lines
    )


def _x_threat_lines(key: str, geometry: optimized_solver.Geometry) -> int:
    return sum(
        sum(key[index] == "X" for index in line) >= 2
        for line in geometry.valid_lines
    )


def candidate_features(
    parent_key: str,
    child_key: str,
    segment: str,
    geometry: optimized_solver.Geometry,
) -> CandidateFeatures:
    """Build model features for a parent → child compressed transition."""
    parent_heuristic = float(optimized_solver._heuristic(parent_key, geometry))
    child_heuristic = float(optimized_solver._heuristic(child_key, geometry))
    pushed_piece = _pushed_piece(parent_key, child_key)
    child_pruned = optimized_solver._pruned(child_key, geometry)

    return CandidateFeatures(
        parent_heuristic=parent_heuristic,
        child_heuristic=child_heuristic,
        heuristic_delta=parent_heuristic - child_heuristic,
        segment_length=len(segment),
        pushed_o=1 if pushed_piece == "O" else 0,
        pushed_x=1 if pushed_piece == "X" else 0,
        walk_only=1 if pushed_piece is None else 0,
        child_solved=1 if optimized_solver._is_solved(child_key, geometry) else 0,
        child_lost=1 if optimized_solver._is_lost(child_key, geometry) else 0,
        child_pruned=1 if child_pruned else 0,
        useful_line_occupancy=_useful_line_occupancy(child_key, geometry),
        x_threat_lines=_x_threat_lines(child_key, geometry),
    )


def candidate_row(
    *,
    parent_key: str,
    child_key: str,
    segment: str,
    geometry: optimized_solver.Geometry,
    label: int | None = None,
    board_id: str | None = None,
    depth: int | None = None,
    remaining_cost: int | None = None,
) -> dict[str, Any]:
    """Return a JSONL-friendly row for one candidate child path."""
    row: dict[str, Any] = {
        "board_id": board_id,
        "depth": depth,
        "parent_key": parent_key,
        "child_key": child_key,
        "segment": segment,
        "label": label,
        "remaining_cost": remaining_cost,
        "features": candidate_features(parent_key, child_key, segment, geometry).to_dict(),
    }
    return row
