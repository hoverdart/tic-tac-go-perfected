"""Independent verifier for push-solver keystroke solutions."""

from __future__ import annotations

from dataclasses import dataclass

from solver.board_utils import normalize_board
from solver.push_solver.core import DIRECTION_BY_MOVE, parse_board


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    final_board: tuple[tuple[str, ...], ...] | None
    error: str | None = None


def _find_player(board: list[list[str]]) -> tuple[int, int] | None:
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell == "U":
                return row_index, col_index
    return None


def _has_win(board: list[list[str]]) -> bool:
    static, _state, _normalized, _player = parse_board(board)
    for line in static.win_lines:
        useful = 0
        for index in line:
            row, col = static.coord(index)
            if board[row][col] in {"U", "O"}:
                useful += 1
        if useful == 3:
            return True
    return False


def _has_x_loss(board: list[list[str]]) -> bool:
    static, _state, _normalized, _player = parse_board(board)
    for line in static.win_lines:
        xs = 0
        for index in line:
            row, col = static.coord(index)
            if board[row][col] == "X":
                xs += 1
        if xs == 3:
            return True
    return False


def verify_solution(board, moves: str | None) -> VerificationResult:
    if moves is None:
        return VerificationResult(ok=False, final_board=None, error="no_moves")

    working = [list(row) for row in normalize_board(board)]
    if _has_x_loss(working):
        return VerificationResult(ok=False, final_board=None, error="x_loss:initial")
    for step, move in enumerate(moves, start=1):
        if move not in DIRECTION_BY_MOVE:
            return VerificationResult(ok=False, final_board=None, error=f"bad_move:{move}")
        player = _find_player(working)
        if player is None:
            return VerificationResult(ok=False, final_board=None, error="missing_player")
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
        else:
            return VerificationResult(ok=False, final_board=None, error=f"bad_cell:{step}")

        if _has_x_loss(working):
            return VerificationResult(ok=False, final_board=None, error=f"x_loss:{step}")

    final_board = tuple(tuple(row) for row in working)
    if not _has_win(working):
        return VerificationResult(ok=False, final_board=final_board, error="not_solved")
    return VerificationResult(ok=True, final_board=final_board)
