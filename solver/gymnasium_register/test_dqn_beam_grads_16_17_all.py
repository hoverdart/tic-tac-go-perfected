"""Run DQN-guided beam search on every grad 16 and 17 training board."""

from __future__ import annotations

import ast
import contextlib
import csv
import io
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch as th
from stable_baselines3 import DQN


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
ALGORITHMS_DIR = REPO_ROOT / "solver" / "algorithms"
TRAINING_BOARDS_PATH = GYM_REGISTER_DIR / "generated_training_boards.py"

GRADS = (16, 17)
BEAM_WIDTH = 5000
BEAM_MAX_DEPTH = 200
REPORT_PATH = GYM_REGISTER_DIR / "dqn_beam_grads_16_17_all_report.csv"
WRITE_CSV = True
ONLY_RERUN_FAILED_FROM_CSV = True
USE_MULTIPROCESSING = True
WORKERS = 6
BOARD_TIMEOUT_SECONDS = 350
BEAM_RESTARTS = 5
RANDOM_TIEBREAK = True
TIEBREAK_NOISE = 0.05
RANDOM_PREFIX_STEPS = [0, 0, 0, 5, 10]
USE_CNN = True
CNN_MODEL_PATH = GYM_REGISTER_DIR / "small_cnn_policy.pt"
CNN_MODEL_ACTION_WEIGHT = 1
RESTART_MODEL_ACTION_WEIGHTS = [0.1, 0.5, 1.0]
TRY_HEURISTIC_FIRST = True
USE_IDDFS = False
IDDFS_DEPTH_STEP = 20

FIELDNAMES = [
    "grad",
    "board_index",
    "board_number",
    "board_line",
    "full_rows",
    "full_cols",
    "active_rows",
    "active_cols",
    "start_soft_locked",
    "solved",
    "beam_move_count",
    "cleaned_move_count",
    "elapsed_seconds",
    "beam_moves",
    "cleaned_moves",
]

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

from beamSearch import beamSearch  # noqa: E402
from iterativeDeepeningDFS import solve as iddfs_solve  # noqa: E402
from generated_training_boards import TRAINING_BOARDS  # noqa: E402
from smallCNN import SmallCNN  # noqa: E402
from run_dqn_grad10_board import (  # noqa: E402
    board_key,
    find_model_path,
    make_env,
    remove_loops,
)
import tic_tac_go_env  # noqa: E402,F401


ACTION_BY_NAME = {"U": 0, "D": 1, "L": 2, "R": 3}


class BoardTimeout(Exception):
    pass


def timeout_handler(_signum, _frame):
    raise BoardTimeout


def active_size(board):
    active_rows = 0
    active_cols = 0
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell != "B":
                active_rows = max(active_rows, row_index + 1)
                active_cols = max(active_cols, col_index + 1)

    return active_rows or len(board), active_cols or len(board[0])


def training_board_line_numbers():
    tree = ast.parse(TRAINING_BOARDS_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TRAINING_BOARDS"
                   for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        line_numbers = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if not isinstance(key_node, ast.Constant):
                continue
            grad = key_node.value
            if grad not in GRADS or not isinstance(value_node, ast.List):
                continue
            line_numbers[grad] = [
                board_node.lineno for board_node in value_node.elts
            ]
        return line_numbers

    raise ValueError(f"Could not find TRAINING_BOARDS in {TRAINING_BOARDS_PATH}")


def replay_moves(board, grad, moves):
    if not moves:
        return False, ""

    env = make_env(board, render_mode=None, grad=grad)
    world = env.unwrapped
    type(world).training_boards = {grad: [board]}
    env.reset(options=grad)

    boards_seen = [board_key(world.board)]
    raw_moves = []
    for move in moves:
        _, _, terminated, truncated, _ = env.step(ACTION_BY_NAME[move])
        raw_moves.append(move)
        boards_seen.append(board_key(world.board))
        if terminated or truncated:
            break

    solved = world.solved(world.board)
    cleaned_moves = remove_loops(raw_moves, boards_seen)
    env.close()
    return solved, cleaned_moves


def start_soft_locked(board, grad):
    env = make_env(board, render_mode=None, grad=grad)
    world = env.unwrapped
    locked = world.softLocked(board)
    env.close()
    return locked


def csv_bool(value):
    return str(value).strip().lower() == "true"


def load_existing_rows():
    if not REPORT_PATH.exists():
        return []

    with REPORT_PATH.open(newline="", encoding="utf-8") as report_file:
        return list(csv.DictReader(report_file))


def save_merged_rows(new_rows):
    existing_rows = load_existing_rows()
    rows_by_key = {
        (int(row["grad"]), int(row["board_index"])): row
        for row in existing_rows
    }
    key_order = [
        (int(row["grad"]), int(row["board_index"]))
        for row in existing_rows
    ]

    for row in new_rows:
        key = (int(row["grad"]), int(row["board_index"]))
        if key not in rows_by_key:
            key_order.append(key)
        rows_by_key[key] = row

    key_order.sort()
    rows = [rows_by_key[key] for key in key_order]
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_search_model(board=None, grad=None):
    if USE_CNN:
        model = SmallCNN()
        model.load_state_dict(th.load(CNN_MODEL_PATH, map_location="cpu"))
        model.eval()
        return model

    model_path = find_model_path()
    if board is None:
        board = tuple(tuple(row) for row in TRAINING_BOARDS[GRADS[0]][0])
    if grad is None:
        grad = GRADS[0]
    env = make_env(board, render_mode=None, grad=grad)
    model = DQN.load(model_path, env=env)
    env.close()
    return model


def make_report_row(model, line_numbers, grad, board_index):
    board = tuple(tuple(row) for row in TRAINING_BOARDS[grad][board_index])
    board_line = line_numbers[grad][board_index]
    full_rows = len(board)
    full_cols = len(board[0])
    active_rows, active_cols = active_size(board)
    soft_locked = start_soft_locked(board, grad)

    restart_seed = (grad * 1_000_000) + (board_index * 10_000)

    def run_search(search_model):
        if USE_IDDFS:
            moves, _final_board, _states_checked = iddfs_solve(
                board,
                model=search_model,
                max_depth=BEAM_MAX_DEPTH,
                depth_step=IDDFS_DEPTH_STEP,
                model_action_weight=CNN_MODEL_ACTION_WEIGHT,
            )
            return moves
        return beamSearch(
            board,
            search_model,
            BEAM_WIDTH,
            BEAM_MAX_DEPTH,
            random_tiebreak=RANDOM_TIEBREAK,
            seed=restart_seed,
            tiebreak_noise=TIEBREAK_NOISE,
            restarts=BEAM_RESTARTS,
            random_prefix_steps=RANDOM_PREFIX_STEPS,
            model_action_weight=CNN_MODEL_ACTION_WEIGHT,
            restart_model_action_weights=RESTART_MODEL_ACTION_WEIGHTS,
        )[0]

    start = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        if TRY_HEURISTIC_FIRST:
            beam_moves = run_search(None)
            solved, cleaned_moves = replay_moves(board, grad, beam_moves)
            if not solved and USE_CNN:
                beam_moves = run_search(model)
                solved, cleaned_moves = replay_moves(board, grad, beam_moves)
        else:
            beam_moves = run_search(model)
            solved, cleaned_moves = replay_moves(board, grad, beam_moves)

    elapsed = time.monotonic() - start

    return {
        "grad": grad,
        "board_index": board_index,
        "board_number": board_index + 1,
        "board_line": board_line,
        "full_rows": full_rows,
        "full_cols": full_cols,
        "active_rows": active_rows,
        "active_cols": active_cols,
        "start_soft_locked": soft_locked,
        "solved": solved,
        "beam_move_count": len(beam_moves),
        "cleaned_move_count": len(cleaned_moves),
        "elapsed_seconds": f"{elapsed:.2f}",
        "beam_moves": beam_moves,
        "cleaned_moves": cleaned_moves,
    }


def make_report_row_worker(task):
    grad, board_index, line_numbers = task
    board = tuple(tuple(row) for row in TRAINING_BOARDS[grad][board_index])
    model = load_search_model(board, grad)
    if BOARD_TIMEOUT_SECONDS is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(BOARD_TIMEOUT_SECONDS)
    try:
        return make_report_row(model, line_numbers, grad, board_index)
    except BoardTimeout:
        full_rows = len(board)
        full_cols = len(board[0])
        active_rows, active_cols = active_size(board)
        return {
            "grad": grad,
            "board_index": board_index,
            "board_number": board_index + 1,
            "board_line": line_numbers[grad][board_index],
            "full_rows": full_rows,
            "full_cols": full_cols,
            "active_rows": active_rows,
            "active_cols": active_cols,
            "start_soft_locked": start_soft_locked(board, grad),
            "solved": False,
            "beam_move_count": 0,
            "cleaned_move_count": 0,
            "elapsed_seconds": f">{BOARD_TIMEOUT_SECONDS}",
            "beam_moves": "TIMEOUT",
            "cleaned_moves": "",
        }
    finally:
        if BOARD_TIMEOUT_SECONDS is not None:
            signal.alarm(0)


def make_report_rows(tasks, model, line_numbers):
    if not USE_MULTIPROCESSING:
        rows = []
        for task_index, (grad, board_index) in enumerate(tasks, start=1):
            row = make_report_row(model, line_numbers, grad, board_index)
            rows.append(row)
            print_progress(task_index, len(tasks), row)
        return rows

    worker_tasks = [
        (grad, board_index, line_numbers)
        for grad, board_index in tasks
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(make_report_row_worker, task)
            for task in worker_tasks
        ]
        for task_index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print_progress(task_index, len(tasks), row)

    rows.sort(key=lambda row: (int(row["grad"]), int(row["board_index"])))
    return rows


def print_progress(index, total, row, prefix=""):
    label = f"{prefix} " if prefix else ""
    print(
        f"{label}{index}/{total} grad={row['grad']} "
        f"board={int(row['board_index']) + 1} line={row['board_line']} "
        f"size={row['active_rows']}x{row['active_cols']} "
        f"solved={row['solved']} soft_locked={row['start_soft_locked']} "
        f"moves={row['beam_move_count']} elapsed={row['elapsed_seconds']}s"
    )


def main():
    model = load_search_model()

    line_numbers = training_board_line_numbers()
    rows = []

    print(f"Use CNN: {USE_CNN}")
    print(f"Model: {CNN_MODEL_PATH if USE_CNN else find_model_path()}")
    if USE_CNN:
        print(f"CNN model action weight: {CNN_MODEL_ACTION_WEIGHT}")
        print(f"Restart CNN weights: {RESTART_MODEL_ACTION_WEIGHTS} then keep last")
    print(f"Try heuristic first: {TRY_HEURISTIC_FIRST}")
    print(f"Report: {REPORT_PATH}")
    print(f"Use IDDFS: {USE_IDDFS}")
    if USE_IDDFS:
        print(f"IDDFS depth step: {IDDFS_DEPTH_STEP}")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Beam max depth: {BEAM_MAX_DEPTH}")
    print(f"Board timeout seconds: {BOARD_TIMEOUT_SECONDS}")
    print(f"Beam restarts: {BEAM_RESTARTS}")
    print(f"Random tiebreak: {RANDOM_TIEBREAK}")
    print(f"Tiebreak noise: {TIEBREAK_NOISE}")
    print(f"Random prefix steps: {RANDOM_PREFIX_STEPS}")
    print(f"Write CSV: {WRITE_CSV}")
    print(f"Only rerun failed rows from CSV: {ONLY_RERUN_FAILED_FROM_CSV}")
    print(f"Use multiprocessing: {USE_MULTIPROCESSING}")
    if USE_MULTIPROCESSING:
        print(f"Workers: {WORKERS}")

    if ONLY_RERUN_FAILED_FROM_CSV:
        existing_rows = load_existing_rows()
        if not existing_rows:
            raise FileNotFoundError(
                f"Cannot rerun failed rows because {REPORT_PATH} does not exist"
            )
        rows_by_key = {
            (int(row["grad"]), int(row["board_index"])): row
            for row in existing_rows
        }
        failed_keys = [
            key for key, row in rows_by_key.items()
            if key[0] in GRADS and not csv_bool(row.get("solved"))
        ]

        print(f"Failed rows to rerun: {len(failed_keys)}")
        for row in make_report_rows(failed_keys, model, line_numbers):
            rows_by_key[(int(row["grad"]), int(row["board_index"]))] = row

        rows = [rows_by_key[(int(row["grad"]), int(row["board_index"]))]
                for row in existing_rows]
        if WRITE_CSV:
            with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
                writer = csv.DictWriter(report_file, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Updated failed rows in report: {REPORT_PATH}")
        else:
            print("CSV write disabled; report was not updated.")
        return

    tasks = [
        (grad, board_index)
        for grad in GRADS
        for board_index, _ in enumerate(TRAINING_BOARDS[grad])
    ]
    rows = make_report_rows(tasks, model, line_numbers)

    for grad in GRADS:
        grad_rows = [row for row in rows if int(row["grad"]) == grad]
        solved_count = sum(csv_bool(row["solved"]) for row in grad_rows)
        print(f"Grad {grad} solved {solved_count}/{len(grad_rows)}")

    if WRITE_CSV:
        save_merged_rows(rows)
        print(f"Updated report: {REPORT_PATH}")
    else:
        print("CSV write disabled; report was not updated.")


if __name__ == "__main__":
    main()
