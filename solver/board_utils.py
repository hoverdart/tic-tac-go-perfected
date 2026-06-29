"""Shared board normalization helpers for solver entry points."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


PIECES = {"X", "O", "U", "B", ""}
IRREGULAR_FILL = "B"


Board = tuple[tuple[str, ...], ...]


def normalize_cell(cell: Any) -> str:
    """Normalize one external cell value into the solver's internal alphabet."""
    if cell is None or cell == ".":
        return ""
    return str(cell).strip().upper()


def normalize_board(board: Sequence[Sequence[Any]]) -> Board:
    """Convert external board data into a rectangular immutable solver board.

    Real puzzle captures can be irregular when non-playable cells are omitted
    from shorter rows. The solver expects rectangular rows, so missing cells are
    padded with barriers (`B`) while explicitly provided empty cells stay empty.
    """
    if not board:
        raise ValueError("Board cannot be empty.")

    row_widths = []
    for row_index, row in enumerate(board):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"Board row {row_index} must be a sequence of cells.")
        row_widths.append(len(row))

    width = max(row_widths, default=0)
    if width == 0:
        raise ValueError("Board rows cannot all be empty.")

    normalized = []
    user_count = 0
    for row_index, row in enumerate(board):
        normalized_row = []
        for col_index, cell in enumerate(row):
            normalized_cell = normalize_cell(cell)
            if normalized_cell not in PIECES:
                raise ValueError(
                    f"Invalid cell {normalized_cell!r} at row {row_index}, "
                    f"col {col_index}."
                )
            if normalized_cell == "U":
                user_count += 1
            normalized_row.append(normalized_cell)

        missing_cells = width - len(normalized_row)
        if missing_cells:
            normalized_row.extend([IRREGULAR_FILL] * missing_cells)
        normalized.append(tuple(normalized_row))

    if user_count != 1:
        raise ValueError(f"Board must contain exactly one U, got {user_count}.")

    return tuple(normalized)


def board_dimensions(board: Sequence[Sequence[Any]]) -> tuple[int, int]:
    """Return row count and widest row width for rectangular or ragged boards."""
    if not board:
        return 0, 0
    return len(board), max((len(row) for row in board), default=0)


def to_wire_board(board: Board) -> list[list[str]]:
    """Convert an immutable internal board to a JSON-serializable board."""
    return [list(row) for row in board]
