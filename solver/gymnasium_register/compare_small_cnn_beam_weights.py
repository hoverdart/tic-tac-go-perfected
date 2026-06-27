"""Compare heuristic/CNN beam weights on the same fixed board sample."""

from __future__ import annotations

import csv
import random
import signal
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch as th


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
ALGORITHMS_DIR = REPO_ROOT / "solver" / "algorithms"
REPORT_PATH = GYM_REGISTER_DIR / "small_cnn_beam_weight_comparison.csv"
MODEL_PATH = GYM_REGISTER_DIR / "small_cnn_policy.pt"

GRADS = (16, 17)
BOARDS_PER_GRAD = 25
BOARD_SAMPLE_SEED = 12345
MODEL_ACTION_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)

BEAM_WIDTH = 5000
BEAM_MAX_DEPTH = 200
BEAM_RESTARTS = 5
RANDOM_TIEBREAK = True
TIEBREAK_NOISE = 0.05
RANDOM_PREFIX_STEPS = [0, 5, 10, 15, 20]
BOARD_TIMEOUT_SECONDS = 350

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import tic_tac_go_env  # noqa: E402,F401
from beamSearch import beamSearch  # noqa: E402
from generated_training_boards import TRAINING_BOARDS  # noqa: E402
from smallCNN import SmallCNN  # noqa: E402


ACTION_BY_NAME = {"U": 0, "D": 1, "L": 2, "R": 3}
FIELDNAMES = [
    "grad",
    "board_index",
    "board_number",
    "weight",
    "solved",
    "status",
    "move_count",
    "elapsed_seconds",
    "moves",
]


class BoardTimeout(Exception):
    pass


def timeout_handler(_signum, _frame):
    raise BoardTimeout


def make_env(board, grad):
    return gym.make(
        "tic_tac_go_env/TicTacWorld-v0",
        length=len(board),
        width=len(board[0]),
        board=board,
        render_mode=None,
        reset_option=grad,
    )


def replay_solved(board, grad, moves):
    if not moves:
        return False

    env = make_env(board, grad)
    world = env.unwrapped
    type(world).training_boards = {grad: [board]}
    env.reset(options=grad)

    for move_name in moves:
        _, _, terminated, truncated, _ = env.step(ACTION_BY_NAME[move_name])
        if terminated or truncated:
            break

    solved = world.solved(world.board)
    env.close()
    return solved


def sample_board_indexes():
    rng = random.Random(BOARD_SAMPLE_SEED)
    samples = {}
    for grad in GRADS:
        board_count = len(TRAINING_BOARDS[grad])
        sample_count = min(BOARDS_PER_GRAD, board_count)
        samples[grad] = sorted(rng.sample(range(board_count), sample_count))
    return samples


def load_model():
    model = SmallCNN()
    model.load_state_dict(th.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def run_beam_with_timeout(board, model, weight, seed):
    if BOARD_TIMEOUT_SECONDS is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(BOARD_TIMEOUT_SECONDS)
    try:
        moves, _transition_data = beamSearch(
            board,
            model,
            BEAM_WIDTH,
            BEAM_MAX_DEPTH,
            random_tiebreak=RANDOM_TIEBREAK,
            seed=seed,
            tiebreak_noise=TIEBREAK_NOISE,
            restarts=BEAM_RESTARTS,
            random_prefix_steps=RANDOM_PREFIX_STEPS,
            model_action_weight=weight,
        )
        return moves, "finished"
    finally:
        if BOARD_TIMEOUT_SECONDS is not None:
            signal.alarm(0)


def main():
    model = load_model()
    samples = sample_board_indexes()
    rows = []
    summary = {
        weight: {"solved": 0, "total": 0, "time": 0.0}
        for weight in MODEL_ACTION_WEIGHTS
    }

    print(f"Report: {REPORT_PATH}")
    print(f"Grads: {GRADS}")
    print(f"Boards per grad: {BOARDS_PER_GRAD}")
    print(f"Weights: {MODEL_ACTION_WEIGHTS}")
    print(f"Board timeout seconds: {BOARD_TIMEOUT_SECONDS}")

    for weight in MODEL_ACTION_WEIGHTS:
        print(f"=== weight {weight} ===")
        for grad in GRADS:
            for board_index in samples[grad]:
                board = tuple(tuple(row) for row in TRAINING_BOARDS[grad][board_index])
                seed = (
                    int(weight * 10_000)
                    + (grad * 1_000_000)
                    + (board_index * 10_000)
                )
                start = time.monotonic()
                try:
                    moves, status = run_beam_with_timeout(
                        board,
                        model,
                        weight,
                        seed,
                    )
                except BoardTimeout:
                    moves = ""
                    status = "timeout"
                elapsed = time.monotonic() - start
                solved = False if status == "timeout" else replay_solved(board, grad, moves)

                rows.append(
                    {
                        "grad": grad,
                        "board_index": board_index,
                        "board_number": board_index + 1,
                        "weight": weight,
                        "solved": solved,
                        "status": status,
                        "move_count": len(moves),
                        "elapsed_seconds": f"{elapsed:.2f}",
                        "moves": moves,
                    }
                )
                summary[weight]["solved"] += int(solved)
                summary[weight]["total"] += 1
                summary[weight]["time"] += elapsed
                print(
                    f"weight={weight} grad={grad} board={board_index + 1} "
                    f"status={status} solved={solved} "
                    f"moves={len(moves)} time={elapsed:.2f}s",
                    flush=True,
                )

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("=== SUMMARY ===")
    for weight, stats in summary.items():
        total = stats["total"]
        solved = stats["solved"]
        avg_time = stats["time"] / total if total else 0.0
        print(
            f"weight={weight}: solved={solved}/{total} "
            f"({solved / total:.1%}), avg_time={avg_time:.2f}s"
        )
    print(f"Wrote: {REPORT_PATH}")


if __name__ == "__main__":
    main()
