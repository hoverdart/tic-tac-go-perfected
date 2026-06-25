"""Add grad 1-15 heuristic-solved state/move pairs to the CNN JSON dataset."""

from __future__ import annotations

import contextlib
import io
import json
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
ALGORITHMS_DIR = REPO_ROOT / "solver" / "algorithms"
OUTPUT_JSON = GYM_REGISTER_DIR / "solved_state_move_pairs.json"

START_GRAD = 9
END_GRAD = 15
USE_MULTIPROCESSING = True
WORKERS = 6
GRADS = range(START_GRAD, END_GRAD + 1)
BEAM_WIDTH = 5000
BEAM_MAX_DEPTH = 200
BEAM_RESTARTS = 3
RANDOM_TIEBREAK = True
TIEBREAK_NOISE = 0.05
RANDOM_PREFIX_STEPS = None
CLEAN_LOOPS = True
SKIP_DUPLICATE_STATE_MOVES = True
BOARD_TIMEOUT_SECONDS = 30

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

from beamSearch import beamSearch  # noqa: E402
from generated_training_boards import TRAINING_BOARDS  # noqa: E402


ACTION_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


class BoardTimeout(Exception):
    pass


def timeout_handler(_signum, _frame):
    raise BoardTimeout


def board_to_rows(board):
    return [" ".join(cell if cell else "." for cell in row) for row in board]


def rows_to_board(rows):
    return tuple(
        tuple("" if cell == "." else cell for cell in row.split())
        for row in rows
    )


def pair_key(pair):
    return tuple(pair["board"]), pair["move"]


def board_key(board):
    return tuple(tuple(row) for row in board)


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


def solved(board):
    for row in range(len(board)):
        for col in range(len(board[row]) - 2):
            if (
                board[row][col] in ("O", "U")
                and board[row][col + 1] in ("O", "U")
                and board[row][col + 2] in ("O", "U")
            ):
                return True

    for row in range(len(board) - 2):
        for col in range(len(board[row])):
            if (
                board[row][col] in ("O", "U")
                and board[row + 1][col] in ("O", "U")
                and board[row + 2][col] in ("O", "U")
            ):
                return True

    return False


def remove_loops(start_board, moves):
    clean_moves = []
    clean_boards = [board_key(start_board)]
    board_indexes = {clean_boards[0]: 0}
    current_board = start_board

    for move_name in moves:
        next_board = move(current_board, move_name)
        next_key = board_key(next_board)

        if next_key == clean_boards[-1]:
            current_board = next_board
            continue

        if next_key in board_indexes:
            keep_until = board_indexes[next_key]
            clean_moves = clean_moves[:keep_until]
            clean_boards = clean_boards[: keep_until + 1]
            board_indexes = {
                board: index for index, board in enumerate(clean_boards)
            }
            current_board = next_board
            continue

        clean_moves.append(move_name)
        clean_boards.append(next_key)
        board_indexes[next_key] = len(clean_boards) - 1
        current_board = next_board

    return "".join(clean_moves)


def pairs_from_moves(start_board, moves):
    pairs = []
    board = start_board

    for move_name in moves:
        next_board = move(board, move_name)
        if next_board == board:
            raise ValueError(f"Move {move_name!r} did not change board")

        pairs.append(
            {
                "board": board_to_rows(board),
                "move": move_name,
            }
        )
        board = next_board

    return pairs, board


def load_existing_pairs():
    if not OUTPUT_JSON.exists():
        return []
    return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))


def save_pairs(output_pairs):
    OUTPUT_JSON.write_text(json.dumps(output_pairs, indent=2), encoding="utf-8")


def solve_board(board, grad, board_index):
    seed = (grad * 1_000_000) + (board_index * 10_000)
    if BOARD_TIMEOUT_SECONDS is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(BOARD_TIMEOUT_SECONDS)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            moves, _transition_data = beamSearch(
                board,
                None,
                BEAM_WIDTH,
                BEAM_MAX_DEPTH,
                random_tiebreak=RANDOM_TIEBREAK,
                seed=seed,
                tiebreak_noise=TIEBREAK_NOISE,
                restarts=BEAM_RESTARTS,
                random_prefix_steps=RANDOM_PREFIX_STEPS,
            )
    finally:
        if BOARD_TIMEOUT_SECONDS is not None:
            signal.alarm(0)
    return moves


def process_board(task):
    grad, board_index, board_count, board = task
    board = tuple(tuple(row) for row in board)

    try:
        moves = solve_board(board, grad, board_index)
    except BoardTimeout:
        return {
            "status": "timeout",
            "grad": grad,
            "board_index": board_index,
            "board_count": board_count,
            "message": f"TIMEOUT after {BOARD_TIMEOUT_SECONDS}s",
            "pairs": [],
            "moves": "",
        }

    if not moves:
        return {
            "status": "failed",
            "grad": grad,
            "board_index": board_index,
            "board_count": board_count,
            "message": "FAILED",
            "pairs": [],
            "moves": "",
        }

    if CLEAN_LOOPS:
        moves = remove_loops(board, moves)

    try:
        pairs, final_board = pairs_from_moves(board, moves)
    except ValueError as error:
        return {
            "status": "failed",
            "grad": grad,
            "board_index": board_index,
            "board_count": board_count,
            "message": str(error),
            "pairs": [],
            "moves": moves,
        }

    if not solved(final_board):
        return {
            "status": "failed",
            "grad": grad,
            "board_index": board_index,
            "board_count": board_count,
            "message": f"moves did not solve after replay ({len(moves)} moves)",
            "pairs": [],
            "moves": moves,
        }

    return {
        "status": "solved",
        "grad": grad,
        "board_index": board_index,
        "board_count": board_count,
        "message": "solved",
        "pairs": pairs,
        "moves": moves,
    }


def process_grad_boards(grad, boards):
    tasks = [
        (grad, board_index, len(boards), board)
        for board_index, board in enumerate(boards)
    ]

    if not USE_MULTIPROCESSING:
        results = []
        for task in tasks:
            result = process_board(task)
            print_worker_result(result)
            results.append(result)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(process_board, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print_worker_result(result)
            results.append(result)

    results.sort(key=lambda result: result["board_index"])
    return results


def print_worker_result(result):
    grad = result["grad"]
    board_index = result["board_index"]
    board_count = result["board_count"]

    if result["status"] == "solved":
        print(
            f"finished grad {grad} board {board_index + 1}/{board_count}: "
            f"solved moves={len(result['moves'])} raw_pairs={len(result['pairs'])}",
            flush=True,
        )
        return

    print(
        f"finished grad {grad} board {board_index + 1}/{board_count}: "
        f"{result['message']}",
        flush=True,
    )


def main():
    output_pairs = load_existing_pairs()
    seen = {pair_key(pair) for pair in output_pairs}
    existing_count = len(output_pairs)
    solved_count = 0
    failed_count = 0
    added_count = 0

    for grad in GRADS:
        boards = TRAINING_BOARDS.get(grad, [])
        grad_solved = 0
        grad_failed = 0
        grad_added = 0

        for result in process_grad_boards(grad, boards):
            board_index = result["board_index"]

            if result["status"] != "solved":
                grad_failed += 1
                failed_count += 1
                continue

            new_pairs = 0
            for pair in result["pairs"]:
                key = pair_key(pair)
                if SKIP_DUPLICATE_STATE_MOVES and key in seen:
                    continue
                output_pairs.append(pair)
                seen.add(key)
                new_pairs += 1

            grad_solved += 1
            solved_count += 1
            grad_added += new_pairs
            added_count += new_pairs
            print(
                f"deduped grad {grad} board {board_index + 1}: "
                f"added_pairs={new_pairs}"
            )

        print(
            f"grad {grad} summary: solved={grad_solved}, "
            f"failed={grad_failed}, added_pairs={grad_added}"
        )
        save_pairs(output_pairs)
        print(f"Saved progress after grad {grad}: {OUTPUT_JSON}")

    save_pairs(output_pairs)
    print(f"Existing pairs: {existing_count}")
    print(f"New pairs added: {added_count}")
    print(f"Total pairs: {len(output_pairs)}")
    print(f"Boards solved: {solved_count}")
    print(f"Boards failed: {failed_count}")
    print(f"Wrote: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
