"""Run the saved DQN model on a random graduation-10 training board."""

from __future__ import annotations

import ast
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
ALGORITHMS_DIR = REPO_ROOT / "solver" / "algorithms"
TRAINING_BOARDS_PATH = GYM_REGISTER_DIR / "generated_training_boards.py"
EVAL_BOARDS_PATH = GYM_REGISTER_DIR / "generated_eval_boards.py"
MAX_STEPS = 200
DETERMINISTIC = False
REPLAY_DELAY_SECONDS = 0.35
FINAL_HOLD_SECONDS = 3.0

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import tic_tac_go_env  # noqa: E402,F401
from beamSearch import beamSearch  # noqa: E402
try:
    from generated_eval_boards import EVAL_BOARDS  # noqa: E402
except ImportError:
    EVAL_BOARDS = {}
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


def board_line_numbers(board_file_path, variable_name, grad):
    tree = ast.parse(board_file_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        for key_node, value_node in zip(node.value.keys, node.value.values):
            if not isinstance(key_node, ast.Constant) or key_node.value != grad:
                continue
            if not isinstance(value_node, ast.List):
                break
            return [board_node.lineno for board_node in value_node.elts]

    raise ValueError(f"Could not find grad {grad} in {board_file_path}")


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


def replay_cleaned_path(env, world, cleaned_moves, action_by_name, grad):
    print("=== CLEANED PATH REPLAY ===")
    env.reset(options=grad)
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


def make_env(board, render_mode, grad):
    return gym.make(
        "tic_tac_go_env/TicTacWorld-v0",
        length=len(board),
        width=len(board[0]),
        board=board,
        render_mode=render_mode,
        reset_option=grad,
    )


def main():
    use_eval_boards = False
    use_beam_search = True
    debug_beam_search = True
    beam_width = 5000
    beam_max_depth = 200
    beam_restarts = 5
    random_tiebreak = True
    tiebreak_noise = 0.05
    grad = 16
    seed = 58
    #hard one: 3502721434

    model_path = find_model_path()
    active_seed = seed if seed is not None else random.randrange(2**32)
    rng = random.Random(active_seed)
    board_pool = EVAL_BOARDS if use_eval_boards and grad in EVAL_BOARDS else TRAINING_BOARDS
    board_source = "eval" if board_pool is EVAL_BOARDS else "training"
    board_file_path = EVAL_BOARDS_PATH if board_source == "eval" else TRAINING_BOARDS_PATH
    board_variable_name = "EVAL_BOARDS" if board_source == "eval" else "TRAINING_BOARDS"
    boards = board_pool[grad]
    board_index = rng.randrange(len(boards))
    board_line = board_line_numbers(board_file_path, board_variable_name, grad)[board_index]
    board = tuple(tuple(row) for row in boards[board_index])

    env = make_env(board, render_mode=None, grad=grad)
    world = env.unwrapped

    model = DQN.load(model_path, env=env)

    # The env's reset samples through the class training-board cache, so pin it
    # to this script's random board before reset initializes locations.
    type(world).training_boards = {grad: [board]}
    obs, _ = env.reset(options=grad)
    start_soft_locked = world.softLocked(world.board)

    print(f"Model: {model_path}")
    print(f"Board source: {board_source}, grad {grad}")
    print(f"Board number: {board_index + 1} / {len(boards)}")
    print(f"Board file line: {board_file_path}:{board_line}")
    print(f"Seed: {active_seed}")
    print(f"Use beam search: {use_beam_search}")
    print("=== START BOARD ===")
    print_board(world.board)
    print(f"Start board soft locked: {start_soft_locked}")
    # if start_soft_locked:
    #     print("Not trying to solve because the starting board is soft locked.")
    #     return

    action_names = {0: "U", 1: "D", 2: "L", 3: "R"}
    action_by_name = {name: action for action, name in action_names.items()}

    if use_beam_search:
        print("=== BEAM SEARCH ===")
        print(f"Beam width: {beam_width}")
        print(f"Max depth: {beam_max_depth}")
        print(f"Beam restarts: {beam_restarts}")
        print(f"Random tiebreak: {random_tiebreak}")
        print(f"Tiebreak noise: {tiebreak_noise}")
        print(f"Debug beam search: {debug_beam_search}")
        beam_moves, transition_data = beamSearch(
            board,
            model,
            beam_width,
            beam_max_depth,
            debug=debug_beam_search,
            random_tiebreak=random_tiebreak,
            seed=active_seed,
            tiebreak_noise=tiebreak_noise,
            restarts=beam_restarts,
        )
        print(f"Beam moves: {beam_moves}")
        print(f"Returned transitions: {len(transition_data)}")
        if beam_moves:
            replay_env = make_env(board, render_mode="human", grad=grad)
            replay_world = replay_env.unwrapped
            type(replay_world).training_boards = {grad: [board]}
            replay_cleaned_path(
                replay_env,
                replay_world,
                beam_moves,
                action_by_name,
                grad,
            )
            replay_env.close()
        env.close()
        return

    attempt = 1

    while True:
        if attempt > 1:
            obs, _ = env.reset(options=grad)
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
                    replay_env = make_env(board, render_mode="human", grad=grad)
                    replay_world = replay_env.unwrapped
                    type(replay_world).training_boards = {grad: [board]}
                    replay_cleaned_path(
                        replay_env,
                        replay_world,
                        cleaned_moves,
                        action_by_name,
                        grad,
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
