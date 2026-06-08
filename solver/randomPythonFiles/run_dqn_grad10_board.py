"""Run the saved DQN model on a random graduation-10 training board."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
MAX_STEPS = 100
DETERMINISTIC = False
REPLAY_DELAY_SECONDS = 0.35
FINAL_HOLD_SECONDS = 3.0

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))

import tic_tac_go_env  # noqa: E402,F401
from generated_training_boards import TRAINING_BOARDS  # noqa: E402


def find_model_path():
    preferred = REPO_ROOT / "dqn_tic_tac_go.zip"
    if preferred.exists():
        return preferred

    matches = sorted(REPO_ROOT.glob("dqn_tic_tac_go*.zip"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not find dqn_tic_tac_go*.zip in the repo root")


def print_board(board):
    for row in board:
        print(" ".join(cell if cell else "." for cell in row))
    print()


def board_key(board):
    return tuple(tuple(row) for row in board)


def remove_loops(moves, boards):
    """Remove moves that revisit an earlier board state."""
    clean_moves = []
    clean_boards = [board_key(boards[0])]
    board_indexes = {clean_boards[0]: 0}

    for move, board_after_move in zip(moves, boards[1:]):
        next_board = board_key(board_after_move)
        current_board = clean_boards[-1]

        if next_board == current_board:
            continue

        if next_board in board_indexes:
            keep_until = board_indexes[next_board]
            clean_moves = clean_moves[:keep_until]
            clean_boards = clean_boards[: keep_until + 1]
            board_indexes = {
                board: index for index, board in enumerate(clean_boards)
            }
            continue

        clean_moves.append(move)
        clean_boards.append(next_board)
        board_indexes[next_board] = len(clean_boards) - 1

    return "".join(clean_moves)


def replay_cleaned_path(env, world, cleaned_moves, action_by_name):
    print("=== CLEANED PATH REPLAY ===")
    env.reset(options=10)
    print("Cleaned step 0: start")
    print_board(world.board)
    time.sleep(REPLAY_DELAY_SECONDS)

    for step, move in enumerate(cleaned_moves, start=1):
        _, reward, terminated, truncated, _ = env.step(action_by_name[move])
        print(f"Cleaned step {step}: {move} reward={reward:.3f}")
        print_board(world.board)
        time.sleep(REPLAY_DELAY_SECONDS)

        if terminated or truncated:
            break

    print("=== CLEANED PATH END ===")
    print(f"Solved after replay: {world.solved(world.board)}")
    time.sleep(FINAL_HOLD_SECONDS)


def make_env(board, render_mode):
    return gym.make(
        "tic_tac_go_env/TicTacWorld-v0",
        length=len(board),
        width=len(board[0]),
        board=board,
        render_mode=render_mode,
        reset_option=10,
    )


def main():
    model_path = find_model_path()
    boards = TRAINING_BOARDS[4]
    board = tuple(tuple(row) for row in random.choice(boards))

    env = make_env(board, render_mode=None)
    world = env.unwrapped

    model = DQN.load(model_path, env=env)

    # The env's reset samples through the class training-board cache, so pin it
    # to this script's random board before reset initializes locations.
    type(world).training_boards = {10: [board]}
    obs, _ = env.reset(options=10)
    start_soft_locked = world.softLocked(world.board)

    print(f"Model: {model_path}")
    print("=== START BOARD ===")
    print_board(world.board)
    print(f"Start board soft locked: {start_soft_locked}")
    if start_soft_locked:
        print("Not trying to solve because the starting board is soft locked.")
        return

    action_names = {0: "U", 1: "D", 2: "L", 3: "R"}
    action_by_name = {name: action for action, name in action_names.items()}
    attempt = 1

    while True:
        if attempt > 1:
            obs, _ = env.reset(options=10)
        moves = []
        boards_seen = [board_key(world.board)]
        total_reward = 0.0

        print(f"=== ATTEMPT {attempt} ===")

        for step in range(1, MAX_STEPS + 1):
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            action_int = int(action)
            moves.append(action_names[action_int])

            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)

            print(f"Step {step}: {action_names[action_int]} reward={reward:.3f}")
            print_board(world.board)
            boards_seen.append(board_key(world.board))

            if terminated or truncated:
                solved = world.solved(world.board)
                lost = world.lostCheck(world.board)
                soft_locked = world.softLocked(world.board)
                print("=== ATTEMPT FINISHED ===")
                print(f"Solved: {solved}")
                print(f"Lost: {lost}")
                print(f"Soft locked: {soft_locked}")
                print(f"Terminated: {terminated}")
                print(f"Truncated: {truncated}")
                print(f"Moves: {''.join(moves)}")
                print(f"Total reward: {total_reward:.3f}")
                print(f"Info: {info}")

                if solved:
                    cleaned_moves = remove_loops(moves, boards_seen)
                    print("=== SOLVED ===")
                    print(f"Attempts: {attempt}")
                    print(f"Raw moves: {''.join(moves)}")
                    print(f"Cleaned moves: {cleaned_moves}")
                    print(f"Removed moves: {len(moves) - len(cleaned_moves)}")
                    replay_env = make_env(board, render_mode="human")
                    replay_world = replay_env.unwrapped
                    type(replay_world).training_boards = {10: [board]}
                    replay_cleaned_path(
                        replay_env,
                        replay_world,
                        cleaned_moves,
                        action_by_name,
                    )
                    replay_env.close()
                    return

                print("Retrying same board.")
                break
        else:
            print("=== ATTEMPT STOPPED ===")
            print(f"Reached MAX_STEPS={MAX_STEPS} without termination.")
            print(f"Moves: {''.join(moves)}")
            print(f"Total reward: {total_reward:.3f}")
            print("Retrying same board.")

        attempt += 1


if __name__ == "__main__":
    main()
