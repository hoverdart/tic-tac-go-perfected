"""Run the saved SmallCNN policy on one selected training/eval board."""

from __future__ import annotations

import ast
import random
import signal
import sys
from pathlib import Path

import gymnasium as gym
import torch as th


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
ALGORITHMS_DIR = REPO_ROOT / "solver" / "algorithms"
TRAINING_BOARDS_PATH = GYM_REGISTER_DIR / "generated_training_boards.py"
EVAL_BOARDS_PATH = GYM_REGISTER_DIR / "generated_eval_boards.py"

GRAD = 17
SEED = 21
USE_EVAL_BOARDS = False
USE_BEAM_SEARCH = True
DEBUG_BEAM_SEARCH = True
BEAM_WIDTH = 5000
BEAM_MAX_DEPTH = 200
BEAM_RESTARTS = 5
RANDOM_TIEBREAK = True
TIEBREAK_NOISE = 0.05
RANDOM_PREFIX_STEPS = [0, 5, 10, 15, 20]
BEAM_TIMEOUT_SECONDS = 350
MAX_STEPS = 200
MODEL_PATH = GYM_REGISTER_DIR / "small_cnn_policy.pt"

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import tic_tac_go_env  # noqa: E402,F401
from beamSearch import beamSearch  # noqa: E402
from generated_training_boards import TRAINING_BOARDS  # noqa: E402
try:
    from generated_eval_boards import EVAL_BOARDS  # noqa: E402
except ImportError:
    EVAL_BOARDS = {}
from smallCNN import SmallCNN  # noqa: E402


ACTION_NAMES = {0: "U", 1: "D", 2: "L", 3: "R"}
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


class BeamTimeout(Exception):
    pass


def timeout_handler(_signum, _frame):
    raise BeamTimeout


def print_board(board):
    for row in board:
        print(" ".join(cell if cell else "." for cell in row))
    print()


def board_to_rows(board):
    return [" ".join(cell if cell else "." for cell in row) for row in board]


def board_key(board):
    return tuple(tuple(row) for row in board)


def find_user(board):
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell == "U":
                return row_index, col_index
    return None


def simulate_action(board, action):
    row_change, col_change = ACTION_DELTAS[action]
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


def lost_check(board):
    for row in range(len(board)):
        for col in range(len(board[row]) - 2):
            if (
                board[row][col] == "X"
                and board[row][col + 1] == "X"
                and board[row][col + 2] == "X"
            ):
                return True

    for row in range(len(board) - 2):
        for col in range(len(board[row])):
            if (
                board[row][col] == "X"
                and board[row + 1][col] == "X"
                and board[row + 2][col] == "X"
            ):
                return True

    return False


def choose_safe_action(board, logits, visited_boards):
    ranked_actions = [
        int(action)
        for action in th.argsort(logits, descending=True).tolist()
    ]
    best_changing_action = None

    for action in ranked_actions:
        next_board = simulate_action(board, action)
        if next_board == board:
            continue
        if best_changing_action is None:
            best_changing_action = action
        if lost_check(next_board):
            continue
        if board_key(next_board) in visited_boards:
            continue
        return action, False

    if best_changing_action is not None:
        return best_changing_action, True

    return None, True


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


def make_env(board, render_mode, grad):
    return gym.make(
        "tic_tac_go_env/TicTacWorld-v0",
        length=len(board),
        width=len(board[0]),
        board=board,
        render_mode=render_mode,
        reset_option=grad,
    )


def replay_moves(board, moves, grad):
    replay_env = make_env(board, render_mode=None, grad=grad)
    replay_world = replay_env.unwrapped
    type(replay_world).training_boards = {grad: [board]}
    replay_env.reset(options=grad)

    action_by_name = {name: action for action, name in ACTION_NAMES.items()}
    total_reward = 0.0
    print("=== BEAM PATH REPLAY ===")
    print_board(replay_world.board)
    for step, move_name in enumerate(moves, start=1):
        _, reward, terminated, truncated, _ = replay_env.step(action_by_name[move_name])
        total_reward += float(reward)
        print(f"Beam step {step}: {move_name} reward={reward:.3f}")
        print_board(replay_world.board)
        if terminated or truncated:
            break

    print("=== BEAM PATH END ===")
    print(f"Solved: {replay_world.solved(replay_world.board)}")
    print(f"Lost: {replay_world.lostCheck(replay_world.board)}")
    print(f"Moves: {moves}")
    print(f"Total reward: {total_reward:.3f}")
    replay_env.close()


def main():
    active_seed = SEED if SEED is not None else random.randrange(2**32)
    rng = random.Random(active_seed)

    board_pool = EVAL_BOARDS if USE_EVAL_BOARDS and GRAD in EVAL_BOARDS else TRAINING_BOARDS
    board_source = "eval" if board_pool is EVAL_BOARDS else "training"
    board_file_path = EVAL_BOARDS_PATH if board_source == "eval" else TRAINING_BOARDS_PATH
    board_variable_name = "EVAL_BOARDS" if board_source == "eval" else "TRAINING_BOARDS"
    boards = board_pool[GRAD]
    board_index = rng.randrange(len(boards))
    board_line = board_line_numbers(board_file_path, board_variable_name, GRAD)[board_index]
    board = tuple(tuple(row) for row in boards[board_index])

    model = SmallCNN()
    model.load_state_dict(th.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    env = make_env(board, render_mode=None, grad=GRAD)
    world = env.unwrapped
    type(world).training_boards = {GRAD: [board]}
    env.reset(options=GRAD)

    print(f"Model: {MODEL_PATH}")
    print(f"Board source: {board_source}, grad {GRAD}")
    print(f"Board number: {board_index + 1} / {len(boards)}")
    print(f"Board file line: {board_file_path}:{board_line}")
    print(f"Seed: {active_seed}")
    print(f"Use beam search: {USE_BEAM_SEARCH}")
    print("=== START BOARD ===")
    print_board(world.board)

    if USE_BEAM_SEARCH:
        print("=== BEAM SEARCH ===")
        print(f"Beam width: {BEAM_WIDTH}")
        print(f"Max depth: {BEAM_MAX_DEPTH}")
        print(f"Beam restarts: {BEAM_RESTARTS}")
        print(f"Random tiebreak: {RANDOM_TIEBREAK}")
        print(f"Tiebreak noise: {TIEBREAK_NOISE}")
        print(f"Random prefix steps: {RANDOM_PREFIX_STEPS}")
        print(f"Beam timeout seconds: {BEAM_TIMEOUT_SECONDS}")
        print("CNN beam scoring: heuristic + 0.5 * cnn_action_logit")
        print(f"Debug beam search: {DEBUG_BEAM_SEARCH}")
        if BEAM_TIMEOUT_SECONDS is not None:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(BEAM_TIMEOUT_SECONDS)
        try:
            beam_moves, transition_data = beamSearch(
                board,
                model,
                BEAM_WIDTH,
                BEAM_MAX_DEPTH,
                debug=DEBUG_BEAM_SEARCH,
                random_tiebreak=RANDOM_TIEBREAK,
                seed=active_seed,
                tiebreak_noise=TIEBREAK_NOISE,
                restarts=BEAM_RESTARTS,
                random_prefix_steps=RANDOM_PREFIX_STEPS,
            )
        except BeamTimeout:
            print("=== BEAM TIMEOUT ===")
            print(f"Seed: {active_seed}")
            print(f"Timed out after {BEAM_TIMEOUT_SECONDS}s")
            env.close()
            return
        finally:
            if BEAM_TIMEOUT_SECONDS is not None:
                signal.alarm(0)
        print(f"Beam moves: {beam_moves}")
        print(f"Returned transitions: {len(transition_data)}")
        if beam_moves:
            replay_moves(board, beam_moves, GRAD)
        env.close()
        return

    moves = []
    visited_boards = {board_key(world.board)}
    total_reward = 0.0
    for step in range(1, MAX_STEPS + 1):
        board_rows = board_to_rows(world.board)
        x = model.get_obs(board_rows).unsqueeze(0)

        with th.no_grad():
            logits = model(x)[0]
            action, used_fallback = choose_safe_action(
                board_key(world.board),
                logits,
                visited_boards,
            )

        if action is None:
            print("=== STOPPED ===")
            print(f"Seed: {active_seed}")
            print("No action changes the board.")
            print(f"Moves: {''.join(moves)}")
            print(f"Total reward: {total_reward:.3f}")
            break

        move_name = ACTION_NAMES[action]
        moves.append(move_name)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        visited_boards.add(board_key(world.board))

        probs = th.softmax(logits, dim=0)
        prob_text = ", ".join(
            f"{ACTION_NAMES[index]}={float(probs[index]):.2f}"
            for index in range(4)
        )
        print(
            f"Step {step}: {move_name} reward={reward:.3f} "
            f"fallback={used_fallback} "
            f"logits={[round(float(v), 3) for v in logits]} probs=[{prob_text}]"
        )
        print_board(world.board)

        if terminated or truncated:
            print("=== FINISHED ===")
            print(f"Seed: {active_seed}")
            print(f"Solved: {world.solved(world.board)}")
            print(f"Lost: {world.lostCheck(world.board)}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Moves: {''.join(moves)}")
            print(f"Total reward: {total_reward:.3f}")
            break
    else:
        print("=== STOPPED ===")
        print(f"Seed: {active_seed}")
        print(f"Reached MAX_STEPS={MAX_STEPS}")
        print(f"Moves: {''.join(moves)}")
        print(f"Total reward: {total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
