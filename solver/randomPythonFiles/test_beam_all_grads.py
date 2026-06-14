"""Run DQN-guided beam search on 50 boards from every available graduation."""

from __future__ import annotations

import contextlib
import csv
import io
import random
import sys
from pathlib import Path

from stable_baselines3 import DQN


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
ALGORITHMS_DIR = REPO_ROOT / "solver" / "algorithms"

RUNS_PER_GRAD = 50
USE_EVAL_BOARDS = True
START_FROM_GRAD = 16
RANDOM_SEED = 5050
BEAM_WIDTH = 1000
BEAM_MAX_DEPTH = 80
REPORT_PATH = GYM_REGISTER_DIR / "beam_all_grads_50_report.csv"

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

from beamSearch import beamSearch  # noqa: E402
from generated_training_boards import TRAINING_BOARDS  # noqa: E402
try:
    from generated_eval_boards import EVAL_BOARDS  # noqa: E402
except ImportError:
    EVAL_BOARDS = {}

from run_dqn_grad10_board import find_model_path, make_env  # noqa: E402
import tic_tac_go_env  # noqa: E402,F401


def active_size(board):
    active_rows = 0
    active_cols = 0
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell != "B":
                active_rows = max(active_rows, row_index + 1)
                active_cols = max(active_cols, col_index + 1)

    return active_rows or len(board), active_cols or len(board[0])


def choose_boards(boards, count, rng):
    if len(boards) >= count:
        indexes = rng.sample(range(len(boards)), count)
    else:
        indexes = [rng.randrange(len(boards)) for _ in range(count)]

    return [(index, tuple(tuple(row) for row in boards[index])) for index in indexes]


def main():
    rng = random.Random(RANDOM_SEED)
    model_path = find_model_path()
    grads = [
        grad
        for grad in sorted(set(TRAINING_BOARDS) | set(EVAL_BOARDS))
        if grad >= START_FROM_GRAD
    ]
    if not grads:
        raise ValueError(f"No boards found at or above START_FROM_GRAD={START_FROM_GRAD}")

    first_grad = grads[0]
    first_pool = (
        EVAL_BOARDS
        if USE_EVAL_BOARDS and first_grad in EVAL_BOARDS
        else TRAINING_BOARDS
    )
    first_board = tuple(tuple(row) for row in first_pool[first_grad][0])
    first_env = make_env(first_board, render_mode=None, grad=first_grad)
    model = DQN.load(model_path, env=first_env)
    first_env.close()

    rows = []
    print(f"Model: {model_path}")
    print(f"Report: {REPORT_PATH}")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Beam max depth: {BEAM_MAX_DEPTH}")
    print(f"Runs per grad: {RUNS_PER_GRAD}")
    print(f"Start from grad: {START_FROM_GRAD}")

    for grad in grads:
        board_pool = (
            EVAL_BOARDS
            if USE_EVAL_BOARDS and grad in EVAL_BOARDS
            else TRAINING_BOARDS
        )
        board_source = "eval" if board_pool is EVAL_BOARDS else "training"
        selected_boards = choose_boards(board_pool[grad], RUNS_PER_GRAD, rng)
        solved_count = 0

        print(f"=== Grad {grad} ({board_source}) ===")

        for run_number, (board_index, board) in enumerate(selected_boards, start=1):
            full_rows = len(board)
            full_cols = len(board[0])
            active_rows, active_cols = active_size(board)

            env = make_env(board, render_mode=None, grad=grad)
            world = env.unwrapped
            type(world).training_boards = {grad: [board]}
            env.reset(options=grad)
            start_board = tuple(tuple(row) for row in world.board)
            start_soft_locked = world.softLocked(start_board)

            if start_soft_locked:
                beam_moves = ""
                solved = False
            else:
                # beamSearch prints its own progress; keep this batch report compact.
                with contextlib.redirect_stdout(io.StringIO()):
                    beam_moves, _ = beamSearch(
                        start_board,
                        model,
                        BEAM_WIDTH,
                        BEAM_MAX_DEPTH,
                    )
                solved = world.solved(start_board) or bool(beam_moves)

            env.close()

            if solved:
                solved_count += 1

            row = {
                "grad": grad,
                "run": run_number,
                "board_source": board_source,
                "board_index": board_index,
                "full_rows": full_rows,
                "full_cols": full_cols,
                "active_rows": active_rows,
                "active_cols": active_cols,
                "start_soft_locked": start_soft_locked,
                "solved": solved,
                "move_count": len(beam_moves),
                "moves": beam_moves,
            }
            rows.append(row)

            print(
                f"grad={grad:02d} run={run_number:02d} "
                f"size={active_rows}x{active_cols} "
                f"solved={solved} soft_locked={start_soft_locked} "
                f"moves={len(beam_moves)}"
            )

        print(f"Grad {grad} solved {solved_count}/{RUNS_PER_GRAD}")

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
