"""Extract (board_state, correct_move) pairs from solved report CSV rows.

By default this reads the grad 16/17 beam report, uses each solved row's
cleaned_moves sequence, replays it from the matching training board, and writes
one row per pre-move board state.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
INPUT_CSV = GYM_REGISTER_DIR / "dqn_beam_grads_16_17_all_report.csv"
OUTPUT_JSON = GYM_REGISTER_DIR / "solved_state_move_pairs.json"
MOVE_FIELD = "cleaned_moves"
SKIP_DUPLICATE_STATE_MOVES = True

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))

from generated_training_boards import TRAINING_BOARDS  # noqa: E402


ACTION_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

def csv_bool(value):
    return str(value).strip().lower() == "true"


def board_to_rows(board):
    return [" ".join(cell if cell else "." for cell in row) for row in board]


def find_user(board):
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell == "U":
                return row_index, col_index
    return None


def move(board, move_name):
    row_change, col_change = ACTION_DELTAS[move_name]
    user_pos = find_user(board)
    if user_pos is None:
        return board

    user_row, user_col = user_pos
    next_row = user_row + row_change
    next_col = user_col + col_change
    push_row = user_row + (2 * row_change)
    push_col = user_col + (2 * col_change)

    if next_row < 0 or next_row >= len(board):
        return board
    if next_col < 0 or next_col >= len(board[next_row]):
        return board

    next_cell = board[next_row][next_col]
    if next_cell == "B":
        return board

    if next_cell in ("X", "O"):
        if push_row < 0 or push_row >= len(board):
            return board
        if push_col < 0 or push_col >= len(board[push_row]):
            return board
        if board[push_row][push_col] != "":
            return board

    new_board = [list(row) for row in board]
    if next_cell in ("X", "O"):
        new_board[push_row][push_col] = next_cell

    new_board[next_row][next_col] = "U"
    new_board[user_row][user_col] = ""
    return tuple(tuple(row) for row in new_board)


def load_solved_rows():
    with INPUT_CSV.open(newline="", encoding="utf-8") as input_file:
        return [
            row for row in csv.DictReader(input_file)
            if csv_bool(row.get("solved")) and row.get(MOVE_FIELD)
        ]


def extract_pairs():
    output_rows = []
    seen = set()

    for source_row in load_solved_rows():
        grad = int(source_row["grad"])
        board_index = int(source_row["board_index"])
        moves = source_row[MOVE_FIELD].strip()
        board = tuple(tuple(row) for row in TRAINING_BOARDS[grad][board_index])

        for step_index, move_name in enumerate(moves, start=1):
            if move_name not in ACTION_DELTAS:
                raise ValueError(
                    f"Bad move {move_name!r} in grad {grad} board {board_index + 1}"
                )

            key = (board, move_name)
            if not SKIP_DUPLICATE_STATE_MOVES or key not in seen:
                output_rows.append(
                    {
                        "board": board_to_rows(board),
                        "move": move_name,
                    }
                )
                seen.add(key)

            next_board = move(board, move_name)
            if next_board == board:
                raise ValueError(
                    f"Move {move_name!r} did not change board at "
                    f"grad {grad} board {board_index + 1} step {step_index}"
                )
            board = next_board

    return output_rows


def main():
    rows = extract_pairs()
    OUTPUT_JSON.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print(f"Input CSV: {INPUT_CSV}")
    print(f"Move field: {MOVE_FIELD}")
    print(f"Solved state/move pairs: {len(rows)}")
    print(f"Output JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
