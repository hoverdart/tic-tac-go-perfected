"""Independent, linear-time verifier for push-solver keystroke solutions.

The verifier deliberately does not call :func:`parse_board`.  Parsing builds
the solver's reverse-push maps and O-pair deadlock tables, which are useful for
search but unnecessary for replay.  V2 rebuilt those tables after nearly every
keystroke, making verification slower than search on large open boards.
"""

from __future__ import annotations

from dataclasses import dataclass

from solver.board_utils import normalize_board
from solver.push_solver.models import DIRECTION_BY_MOVE


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    final_board: tuple[tuple[str, ...], ...] | None
    error: str | None = None


Line = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def _valid_lines(board: list[list[str]]) -> tuple[Line, ...]:
    rows = len(board)
    cols = len(board[0]) if board else 0
    lines: list[Line] = []
    for row in range(rows):
        for col in range(cols - 2):
            line = ((row, col), (row, col + 1), (row, col + 2))
            if all(board[line_row][line_col] != "B" for line_row, line_col in line):
                lines.append(line)
    for row in range(rows - 2):
        for col in range(cols):
            line = ((row, col), (row + 1, col), (row + 2, col))
            if all(board[line_row][line_col] != "B" for line_row, line_col in line):
                lines.append(line)
    return tuple(lines)


def _line_has_cells(
    board: list[list[str]],
    line: Line,
    cells: frozenset[str],
) -> bool:
    return all(board[row][col] in cells for row, col in line)


def _has_win(board: list[list[str]], lines: tuple[Line, ...] | None = None) -> bool:
    lines = _valid_lines(board) if lines is None else lines
    return any(_line_has_cells(board, line, frozenset({"U", "O"})) for line in lines)


def _has_x_loss(board: list[list[str]], lines: tuple[Line, ...] | None = None) -> bool:
    lines = _valid_lines(board) if lines is None else lines
    return any(_line_has_cells(board, line, frozenset({"X"})) for line in lines)


def verify_solution(board, moves: str | None) -> VerificationResult:
    if moves is None:
        return VerificationResult(ok=False, final_board=None, error="no_moves")

    working = [list(row) for row in normalize_board(board)]
    lines = _valid_lines(working)
    player = next(
        (
            (row_index, col_index)
            for row_index, row in enumerate(working)
            for col_index, cell in enumerate(row)
            if cell == "U"
        ),
        None,
    )
    if player is None:
        return VerificationResult(ok=False, final_board=None, error="missing_player")
    if _has_x_loss(working, lines):
        return VerificationResult(ok=False, final_board=None, error="x_loss:initial")
    for step, move in enumerate(moves, start=1):
        if move not in DIRECTION_BY_MOVE:
            return VerificationResult(ok=False, final_board=None, error=f"bad_move:{move}")
        dr, dc = DIRECTION_BY_MOVE[move]
        row, col = player
        next_row = row + dr
        next_col = col + dc
        if not (0 <= next_row < len(working) and 0 <= next_col < len(working[0])):
            return VerificationResult(ok=False, final_board=None, error=f"off_board:{step}")
        target = working[next_row][next_col]
        if target == "B":
            return VerificationResult(ok=False, final_board=None, error=f"wall:{step}")
        if target == "":
            working[row][col] = ""
            working[next_row][next_col] = "U"
            player = (next_row, next_col)
        elif target in {"O", "X"}:
            push_row = next_row + dr
            push_col = next_col + dc
            if not (0 <= push_row < len(working) and 0 <= push_col < len(working[0])):
                return VerificationResult(ok=False, final_board=None, error=f"push_off_board:{step}")
            if working[push_row][push_col] != "":
                return VerificationResult(ok=False, final_board=None, error=f"blocked_push:{step}")
            working[row][col] = ""
            working[next_row][next_col] = "U"
            working[push_row][push_col] = target
            player = (next_row, next_col)
        else:
            return VerificationResult(ok=False, final_board=None, error=f"bad_cell:{step}")

        if target == "X" and _has_x_loss(working, lines):
            return VerificationResult(ok=False, final_board=None, error=f"x_loss:{step}")

    final_board = tuple(tuple(row) for row in working)
    if not _has_win(working, lines):
        return VerificationResult(ok=False, final_board=final_board, error="not_solved")
    return VerificationResult(ok=True, final_board=final_board)
