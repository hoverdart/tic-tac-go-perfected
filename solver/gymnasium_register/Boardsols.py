import json
import requests
import signal
import sys
import torch as th
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path

# =======================================================
# SET YOUR BOARD RANGE HERE
# =======================================================
START_PUZZLE_NUM = 246
END_PUZZLE_NUM = 341  # None means use the current puzzle number from the anchor/date.

# The true anchor based on your exact confirmation
ANCHOR_PUZZLE_NUM = 271
ANCHOR_DATE_STR = "20260621"  # June 21, 2026

BEAM_TIMEOUT_SECONDS = 350
BEAM_WIDTH = 5000
BEAM_MAX_DEPTH = 200
BEAM_RESTARTS = 5
RANDOM_TIEBREAK = True
TIEBREAK_NOISE = 0.05
RANDOM_PREFIX_STEPS = [0, 0, 0, 5, 10]
CNN_MODEL_ACTION_WEIGHT = 1
RESTART_MODEL_ACTION_WEIGHTS = [0.1, 0.5, 1.0]
USE_MULTIPROCESSING = True
WORKERS = 6
# =======================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ALL_BOARDS_PATH = SCRIPT_DIR / "allBoards.json"
STATE_MOVE_PAIRS_PATH = REPO_ROOT / "solver" / "gymnasium_register" / "solved_state_move_pairs.json"
CNN_MODEL_PATH = REPO_ROOT / "solver" / "gymnasium_register" / "small_cnn_policy.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.algorithms.beamSearch import beamSearch  # noqa: E402
from solver.gymnasium_register.smallCNN import SmallCNN  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CELL_MAP = {
    "-": ".",
    ".": ".",
    "W": "B",
    "B": "B",
    "P": "U",
    "U": "U",
    "X": "X",
    "O": "O",
}
MOVE_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
_CNN_MODEL = None


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_puzzles():
    return json.loads(ALL_BOARDS_PATH.read_text(encoding="utf-8"))


def decode_board(puzzle):
    width = puzzle["width"]
    height = puzzle["height"]
    cells = puzzle["puzzle"]

    board = []
    for row_index in range(height):
        row = cells[row_index * width:(row_index + 1) * width]
        board.append([CELL_MAP.get(cell, cell) for cell in row])
    return board


def board_for_beam(board):
    return tuple(
        tuple("" if cell == "." else cell for cell in row)
        for row in board
    )


def board_to_rows(board):
    return [" ".join("." if cell == "" else cell for cell in row) for row in board]


def rows_to_board(rows):
    return [row.split() for row in rows]


def find_user(board):
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell == "U":
                return row_index, col_index
    raise ValueError("Board has no U piece")


def apply_move(board, move):
    row_delta, col_delta = MOVE_DELTAS[move]
    user_row, user_col = find_user(board)
    next_row = user_row + row_delta
    next_col = user_col + col_delta

    if not (0 <= next_row < len(board) and 0 <= next_col < len(board[next_row])):
        return board
    if board[next_row][next_col] == "B":
        return board

    push_row = next_row + row_delta
    push_col = next_col + col_delta
    target_cell = board[next_row][next_col]

    if target_cell in ("X", "O"):
        if not (0 <= push_row < len(board) and 0 <= push_col < len(board[push_row])):
            return board
        if board[push_row][push_col] != ".":
            return board

    new_board = [row[:] for row in board]
    if target_cell in ("X", "O"):
        new_board[push_row][push_col] = target_cell

    new_board[next_row][next_col] = "U"
    new_board[user_row][user_col] = "."
    return new_board


def solution_moves_for_date(date_str):
    url = f"https://tic-tac-go.com/api/puzzles/{date_str}/solution"
    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code != 200:
        return None, f"HTTP {response.status_code}"

    data = response.json()
    moves = data.get("moves", [])
    if not moves:
        return "", f"No moves found. Server returned: {data}"

    move_string = ""
    for move in moves:
        from_row = move.get("playerFromRow", 0)
        from_col = move.get("playerFromCol", 0)
        to_row = move.get("playerToRow", 0)
        to_col = move.get("playerToCol", 0)

        if to_row < from_row:
            move_string += "U"
        elif to_row > from_row:
            move_string += "D"
        elif to_col < from_col:
            move_string += "L"
        elif to_col > from_col:
            move_string += "R"

    return move_string, None


def load_cnn_model():
    global _CNN_MODEL
    if _CNN_MODEL is not None:
        return _CNN_MODEL

    model = SmallCNN()
    model.load_state_dict(th.load(CNN_MODEL_PATH, map_location="cpu"))
    model.eval()
    _CNN_MODEL = model
    return model


def beam_solve(board, model=None):
    beam_board = board_for_beam(board)
    moves, _transition_data = beamSearch(
        beam_board,
        model,
        BEAM_WIDTH,
        BEAM_MAX_DEPTH,
        restarts=BEAM_RESTARTS,
        random_tiebreak=RANDOM_TIEBREAK,
        tiebreak_noise=TIEBREAK_NOISE,
        random_prefix_steps=RANDOM_PREFIX_STEPS,
        model_action_weight=CNN_MODEL_ACTION_WEIGHT,
        restart_model_action_weights=RESTART_MODEL_ACTION_WEIGHTS,
    )
    return moves


def fallback_solution_moves(board):
    try:
        with time_limit(BEAM_TIMEOUT_SECONDS):
            moves = beam_solve(board, model=None)
        if moves:
            return moves, "heuristic"
    except TimeoutError:
        print(f"  heuristic beam timed out after {BEAM_TIMEOUT_SECONDS}s")

    try:
        model = load_cnn_model()
    except Exception as exc:
        print(f"  could not load CNN model: {exc}")
        return "", None

    try:
        with time_limit(BEAM_TIMEOUT_SECONDS):
            moves = beam_solve(board, model=model)
        if moves:
            return moves, "heuristic+cnn"
    except TimeoutError:
        print(f"  heuristic+cnn beam timed out after {BEAM_TIMEOUT_SECONDS}s")

    return "", None


def load_existing_pairs():
    if not STATE_MOVE_PAIRS_PATH.exists():
        return []
    return json.loads(STATE_MOVE_PAIRS_PATH.read_text(encoding="utf-8"))


def save_pairs(pairs):
    STATE_MOVE_PAIRS_PATH.write_text(json.dumps(pairs, indent=2), encoding="utf-8")


def add_board_solution_pairs(pairs, seen_pairs, board, moves):
    added = 0
    current_board = [row[:] for row in board]

    for move in moves:
        board_rows = board_to_rows(current_board)
        key = (tuple(board_rows), move)
        if key not in seen_pairs:
            pairs.append({"board": board_rows, "move": move})
            seen_pairs.add(key)
            added += 1

        next_board = apply_move(current_board, move)
        if next_board == current_board:
            raise ValueError(f"Move {move} did not change the board")
        current_board = next_board

    return added


def state_move_pairs_for_solution(board, moves):
    pairs = []
    current_board = [row[:] for row in board]

    for move in moves:
        pairs.append({"board": board_to_rows(current_board), "move": move})

        next_board = apply_move(current_board, move)
        if next_board == current_board:
            raise ValueError(f"Move {move} did not change the board")
        current_board = next_board

    return pairs


def process_board(board_num, puzzle):
    anchor_date = datetime.strptime(ANCHOR_DATE_STR, "%Y%m%d")
    days_difference = board_num - ANCHOR_PUZZLE_NUM
    target_date = anchor_date + timedelta(days=days_difference)
    date_str = target_date.strftime("%Y%m%d")

    try:
        board = decode_board(puzzle)
        moves, error = solution_moves_for_date(date_str)
        source = "api"
        if error:
            moves, source = fallback_solution_moves(board)
            if not moves:
                return {
                    "board_num": board_num,
                    "date_str": date_str,
                    "status": "failed",
                    "message": f"{error}; no fallback solution found",
                    "source": None,
                    "moves": "",
                    "pairs": [],
                }

        return {
            "board_num": board_num,
            "date_str": date_str,
            "status": "solved",
            "message": "",
            "source": source,
            "moves": moves,
            "pairs": state_move_pairs_for_solution(board, moves),
        }
    except Exception as exc:
        return {
            "board_num": board_num,
            "date_str": date_str,
            "status": "failed",
            "message": str(exc),
            "source": None,
            "moves": "",
            "pairs": [],
        }


def add_solutions_to_state_move_pairs():
    anchor_date = datetime.strptime(ANCHOR_DATE_STR, "%Y%m%d")

    end_puzzle_num = END_PUZZLE_NUM
    if end_puzzle_num is None:
        end_puzzle_num = ANCHOR_PUZZLE_NUM + (datetime.now() - anchor_date).days

    puzzles = load_puzzles()
    puzzle_by_id = {puzzle["id"]: puzzle for puzzle in puzzles}
    pairs = load_existing_pairs()
    seen_pairs = {
        (tuple(pair["board"]), pair["move"])
        for pair in pairs
    }

    print(f"Loaded {len(puzzles)} boards from {ALL_BOARDS_PATH}")
    print(f"Loaded {len(pairs)} existing state/move pairs")
    print(f"Adding puzzles {START_PUZZLE_NUM} through {end_puzzle_num}\n")

    total_added = 0
    total_solved = 0
    board_jobs = []
    for board_num in range(START_PUZZLE_NUM, end_puzzle_num + 1):
        date_str = (
            anchor_date + timedelta(days=board_num - ANCHOR_PUZZLE_NUM)
        ).strftime("%Y%m%d")
        puzzle = puzzle_by_id.get(date_str)
        if puzzle is None:
            print(f"Board {board_num} ({date_str}): missing from allBoards.json")
            continue
        board_jobs.append((board_num, puzzle))

    print(
        f"Processing {len(board_jobs)} boards "
        f"with {'multiprocessing' if USE_MULTIPROCESSING else 'one process'}"
    )

    def merge_result(result):
        nonlocal total_added, total_solved

        board_num = result["board_num"]
        date_str = result["date_str"]
        if result["status"] != "solved":
            print(f"Board {board_num} ({date_str}): {result['message']}", flush=True)
            return

        added = 0
        for pair in result["pairs"]:
            key = (tuple(pair["board"]), pair["move"])
            if key in seen_pairs:
                continue
            pairs.append(pair)
            seen_pairs.add(key)
            added += 1

        total_added += added
        total_solved += 1
        print(
            f"Board {board_num} ({date_str}): source={result['source']} "
            f"moves={len(result['moves'])} added_pairs={added}",
            flush=True,
        )
        save_pairs(pairs)

    if USE_MULTIPROCESSING:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = [
                executor.submit(process_board, board_num, puzzle)
                for board_num, puzzle in board_jobs
            ]
            for future in as_completed(futures):
                merge_result(future.result())
    else:
        for board_num, puzzle in board_jobs:
            merge_result(process_board(board_num, puzzle))

    save_pairs(pairs)
    print()
    print(f"Solved boards added: {total_solved}")
    print(f"New state/move pairs added: {total_added}")
    print(f"Total state/move pairs now: {len(pairs)}")
    print(f"Saved to {STATE_MOVE_PAIRS_PATH}")


if __name__ == "__main__":
    add_solutions_to_state_move_pairs()
