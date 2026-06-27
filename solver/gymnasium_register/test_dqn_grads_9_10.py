"""Run normal DQN policy tests on 50 grad 10 boards, no beam search."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import torch as th
from stable_baselines3 import DQN


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"

GRADS = (10,)
RUNS_PER_GRAD = 1
USE_EVAL_BOARDS = True
RANDOM_SEED = 78
DETERMINISTIC = False
REPORT_PATH = GYM_REGISTER_DIR / "dqn_grad10_50_report.csv"
STEP_REPORT_PATH = GYM_REGISTER_DIR / "dqn_grad10_50_step_report.csv"

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))

from generated_training_boards import TRAINING_BOARDS  # noqa: E402
try:
    from generated_eval_boards import EVAL_BOARDS  # noqa: E402
except ImportError:
    EVAL_BOARDS = {}

from run_dqn_grad10_board import (  # noqa: E402
    board_key,
    find_model_path,
    make_env,
    remove_loops,
)
import tic_tac_go_env  # noqa: E402,F401


ACTION_NAMES = {0: "U", 1: "D", 2: "L", 3: "R"}


def choose_boards(boards, count, rng):
    if len(boards) >= count:
        indexes = rng.sample(range(len(boards)), count)
    else:
        indexes = [rng.randrange(len(boards)) for _ in range(count)]

    return [(index, tuple(tuple(row) for row in boards[index])) for index in indexes]


def board_to_text(board):
    return "\n".join(" ".join(cell if cell else "." for cell in row) for row in board)


def q_values_for_obs(model, obs):
    obs_tensor = th.tensor(obs, dtype=th.float32, device=model.device).unsqueeze(0)
    with th.no_grad():
        return model.q_net(obs_tensor).cpu().numpy().reshape(-1)


def run_one_board(board, grad, model):
    env = make_env(board, render_mode=None, grad=grad)
    world = env.unwrapped
    type(world).training_boards = {grad: [board]}
    obs, _ = env.reset(options=grad)

    start_board = tuple(tuple(row) for row in world.board)
    start_soft_locked = world.softLocked(start_board)
    moves = []
    boards_seen = [board_key(start_board)]
    total_reward = 0.0
    steps = 0
    solved = False
    lost = False
    soft_locked = start_soft_locked
    terminated = False
    truncated = False
    step_rows = []

    if not start_soft_locked:
        while True:
            board_before = tuple(tuple(row) for row in world.board)
            q_values = q_values_for_obs(model, obs)
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            action_int = int(action)
            moves.append(ACTION_NAMES[action_int])

            obs, reward, terminated, truncated, _ = env.step(action_int)
            board_after = tuple(tuple(row) for row in world.board)
            board_changed = board_after != board_before
            total_reward += float(reward)
            steps += 1
            boards_seen.append(board_key(world.board))

            solved = world.solved(world.board)
            lost = world.lostCheck(world.board)
            soft_locked = world.softLocked(world.board)
            step_rows.append(
                {
                    "step": steps,
                    "board_state": board_to_text(board_before),
                    "q_up": float(q_values[0]),
                    "q_down": float(q_values[1]),
                    "q_left": float(q_values[2]),
                    "q_right": float(q_values[3]),
                    "chosen_action": ACTION_NAMES[action_int],
                    "board_changed": board_changed,
                    "reward": float(reward),
                    "terminated": terminated,
                    "truncated": truncated,
                    "solved_after": solved,
                    "lost_after": lost,
                    "soft_locked_after": soft_locked,
                }
            )

            if terminated or truncated:
                break

    cleaned_moves = remove_loops(moves, boards_seen)
    env.close()

    summary = {
        "solved": solved,
        "start_soft_locked": start_soft_locked,
        "lost": lost,
        "soft_locked": soft_locked,
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps,
        "reward": total_reward,
        "raw_move_count": len(moves),
        "cleaned_move_count": len(cleaned_moves),
        "raw_moves": "".join(moves),
        "cleaned_moves": cleaned_moves,
    }
    return summary, step_rows


def main():
    rng = random.Random(RANDOM_SEED)
    model_path = find_model_path()
    first_pool = EVAL_BOARDS if USE_EVAL_BOARDS and GRADS[0] in EVAL_BOARDS else TRAINING_BOARDS
    first_board = tuple(tuple(row) for row in first_pool[GRADS[0]][0])
    first_env = make_env(first_board, render_mode=None, grad=GRADS[0])
    model = DQN.load(model_path, env=first_env)
    first_env.close()

    rows = []
    step_rows = []
    print(f"Model: {model_path}")
    print(f"Report: {REPORT_PATH}")
    print(f"Step report: {STEP_REPORT_PATH}")
    print(f"Deterministic: {DETERMINISTIC}")

    for grad in GRADS:
        board_pool = EVAL_BOARDS if USE_EVAL_BOARDS and grad in EVAL_BOARDS else TRAINING_BOARDS
        board_source = "eval" if board_pool is EVAL_BOARDS else "training"
        selected_boards = choose_boards(board_pool[grad], RUNS_PER_GRAD, rng)
        solved_count = 0

        print(f"=== Grad {grad} ({board_source}) ===")
        for run_number, (board_index, board) in enumerate(selected_boards, start=1):
            result, board_step_rows = run_one_board(board, grad, model)
            solved_count += int(result["solved"])
            row = {
                "grad": grad,
                "run": run_number,
                "board_source": board_source,
                "board_index": board_index,
                **result,
            }
            rows.append(row)
            for step_row in board_step_rows:
                step_rows.append(
                    {
                        "grad": grad,
                        "run": run_number,
                        "board_source": board_source,
                        "board_index": board_index,
                        **step_row,
                    }
                )
            print(
                f"grad={grad} run={run_number:02d} solved={result['solved']} "
                f"steps={result['steps']} reward={result['reward']:.2f} "
                f"raw={result['raw_move_count']} cleaned={result['cleaned_move_count']} "
                f"truncated={result['truncated']}"
            )

        print(f"Grad {grad} solved {solved_count}/{RUNS_PER_GRAD}")

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if step_rows:
        with STEP_REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=list(step_rows[0]))
            writer.writeheader()
            writer.writerows(step_rows)

    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved step report: {STEP_REPORT_PATH}")


if __name__ == "__main__":
    main()
