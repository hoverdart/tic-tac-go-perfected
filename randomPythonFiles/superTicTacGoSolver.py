import argparse
import datetime
import heapq
import itertools
import sys
import time
import webbrowser
from collections import deque
from functools import lru_cache
from pathlib import Path


DEFAULT_BOARD = (
    ("", "", "X", "", "", ""),
    ("", "O", "", "X", "", ""),
    ("", "B", "", "X", "", "X"),
    ("X", "", "X", "", "B", ""),
    ("", "", "", "X", "O", ""),
    ("", "X", "", "", "", "U"),
)

DIRECTIONS = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)

KEY_MAP = {
    "U": "up",
    "D": "down",
    "L": "left",
    "R": "right",
}

PIECES = {"X", "O", "U", "B", ""}
MOVABLE_PIECES = {"X", "O"}
USEFUL_PIECES = {"O", "U"}


def normalize_board(board):
    """Convert parser or hand-written boards into the solver's immutable format."""
    if not board:
        raise ValueError("Board cannot be empty.")

    width = len(board[0])
    if width == 0:
        raise ValueError("Board rows cannot be empty.")

    normalized = []
    user_count = 0
    for row_index, row in enumerate(board):
        if len(row) != width:
            raise ValueError("Board must be rectangular.")

        normalized_row = []
        for col_index, cell in enumerate(row):
            cell = "" if cell in (None, ".") else str(cell).strip().upper()
            if cell not in PIECES:
                raise ValueError(
                    f"Invalid cell {cell!r} at row {row_index}, col {col_index}."
                )
            if cell == "U":
                user_count += 1
            normalized_row.append(cell)

        normalized.append(tuple(normalized_row))

    if user_count != 1:
        raise ValueError(f"Board must contain exactly one U, got {user_count}.")

    return tuple(normalized)


def board_size(board):
    return len(board), len(board[0])


def in_bounds(board, row, col):
    rows, cols = board_size(board)
    return 0 <= row < rows and 0 <= col < cols


def find_user(board):
    for row, cells in enumerate(board):
        for col, cell in enumerate(cells):
            if cell == "U":
                return row, col

    raise ValueError("Board has no U piece.")


@lru_cache(maxsize=None)
def possible_win_lines(rows, cols):
    """All horizontal and vertical runs of three that can end the game."""
    lines = []

    for row in range(rows):
        for col in range(cols - 2):
            lines.append(((row, col), (row, col + 1), (row, col + 2)))

    for row in range(rows - 2):
        for col in range(cols):
            lines.append(((row, col), (row + 1, col), (row + 2, col)))

    return tuple(lines)


def win_lines_for(board):
    rows, cols = board_size(board)
    return possible_win_lines(rows, cols)


def lost_check(board):
    return any(
        all(board[row][col] == "X" for row, col in line)
        for line in win_lines_for(board)
    )


def solved(board):
    return any(
        all(board[row][col] in USEFUL_PIECES for row, col in line)
        for line in win_lines_for(board)
    )


def dead_board(board):
    useful_piece_count = sum(cell in USEFUL_PIECES for row in board for cell in row)
    if useful_piece_count < 3:
        return True

    # Barriers never move. If every possible win line contains a barrier, there
    # is nowhere left to form O/O/U.
    return all(
        any(board[row][col] == "B" for row, col in line)
        for line in win_lines_for(board)
    )


def piece_can_move(board, row, col):
    for _, row_delta, col_delta in DIRECTIONS:
        push_from_row = row - row_delta
        push_from_col = col - col_delta
        push_to_row = row + row_delta
        push_to_col = col + col_delta

        if not in_bounds(board, push_from_row, push_from_col):
            continue
        if not in_bounds(board, push_to_row, push_to_col):
            continue
        if board[push_from_row][push_from_col] not in ("", "U"):
            continue
        if board[push_to_row][push_to_col] == "":
            return True

    return False


def spot_can_become_user(board, row, col):
    if board[row][col] in ("", "U"):
        return True
    if board[row][col] == "X":
        return piece_can_move(board, row, col)
    return False


def spot_can_become_empty(board, row, col):
    if board[row][col] == "":
        return True
    if board[row][col] == "X":
        return piece_can_move(board, row, col)
    return False


def useful_piece_can_move(board, row, col):
    for _, row_delta, col_delta in DIRECTIONS:
        push_from_row = row - row_delta
        push_from_col = col - col_delta
        push_to_row = row + row_delta
        push_to_col = col + col_delta

        if not in_bounds(board, push_from_row, push_from_col):
            continue
        if not in_bounds(board, push_to_row, push_to_col):
            continue
        if not spot_can_become_user(board, push_from_row, push_from_col):
            continue
        if spot_can_become_empty(board, push_to_row, push_to_col):
            return True

    return False


def soft_locked(board):
    """Cheap prune for boards where both O pieces are separated and trapped."""
    o_locations = [
        (row, col)
        for row, cells in enumerate(board)
        for col, cell in enumerate(cells)
        if cell == "O"
    ]

    if len(o_locations) == 2:
        first_o, second_o = o_locations
        os_are_aligned = first_o[0] == second_o[0] or first_o[1] == second_o[1]
        if not os_are_aligned:
            if not any(
                useful_piece_can_move(board, row, col)
                for row, col in o_locations
            ):
                return True

    return False


def pruned_board(board):
    return lost_check(board) or dead_board(board) or soft_locked(board)


def line_score(board, line, useful_pieces):
    occupied = sum(board[row][col] in USEFUL_PIECES for row, col in line)
    blockers = sum(board[row][col] == "X" for row, col in line)
    distance = 0

    for target_row, target_col in line:
        distance += min(
            abs(piece_row - target_row) + abs(piece_col - target_col)
            for piece_row, piece_col in useful_pieces
        )

    # This is intentionally a greedy estimate, not a proof of optimality. It
    # keeps the search focused on lines that already contain O/U pieces and
    # penalizes X pieces that must be moved away.
    return distance - (occupied * 3) + (blockers * 4)


def heuristic(board):
    if solved(board):
        return 0

    useful_pieces = [
        (row, col)
        for row, cells in enumerate(board)
        for col, cell in enumerate(cells)
        if cell in USEFUL_PIECES
    ]
    if len(useful_pieces) < 3:
        return float("inf")

    scores = [
        line_score(board, line, useful_pieces)
        for line in win_lines_for(board)
        if not any(board[row][col] == "B" for row, col in line)
    ]
    return max(0, min(scores, default=float("inf")))


def reachable_empty_cells(board):
    """Shortest walking path from U to each empty cell without pushing pieces."""
    start = find_user(board)
    queue = deque([(start, "")])
    paths = {start: ""}

    while queue:
        (row, col), path = queue.popleft()

        for move, row_delta, col_delta in DIRECTIONS:
            next_row = row + row_delta
            next_col = col + col_delta
            next_pos = (next_row, next_col)

            if not in_bounds(board, next_row, next_col):
                continue
            if next_pos in paths or board[next_row][next_col] != "":
                continue

            paths[next_pos] = path + move
            queue.append((next_pos, path + move))

    return paths


def move_user_to(board, target_pos):
    user_row, user_col = find_user(board)
    target_row, target_col = target_pos
    if (user_row, user_col) == (target_row, target_col):
        return board

    new_board = [list(row) for row in board]
    new_board[user_row][user_col] = ""
    new_board[target_row][target_col] = "U"
    return tuple(tuple(row) for row in new_board)


def push_from(board, user_pos, move, row_delta, col_delta):
    user_row, user_col = user_pos
    piece_row = user_row + row_delta
    piece_col = user_col + col_delta
    landing_row = user_row + (row_delta * 2)
    landing_col = user_col + (col_delta * 2)

    if not in_bounds(board, piece_row, piece_col):
        return None
    if not in_bounds(board, landing_row, landing_col):
        return None
    if board[piece_row][piece_col] not in MOVABLE_PIECES:
        return None
    if board[landing_row][landing_col] != "":
        return None

    new_board = [list(row) for row in board]
    new_board[user_row][user_col] = ""
    new_board[landing_row][landing_col] = board[piece_row][piece_col]
    new_board[piece_row][piece_col] = "U"
    return tuple(tuple(row) for row in new_board), move


def next_boards(board):
    paths = reachable_empty_cells(board)

    # Walking alone can finish a board because U counts as a useful piece.
    for user_pos, walk_path in paths.items():
        if not walk_path:
            continue
        walked_board = move_user_to(board, user_pos)
        if solved(walked_board):
            yield walked_board, walk_path

    # Collapse long walking runs into a single edge, then branch only on pushes.
    # This searches the meaningful puzzle states instead of every empty step.
    for user_pos, walk_path in paths.items():
        walked_board = move_user_to(board, user_pos)
        for move, row_delta, col_delta in DIRECTIONS:
            pushed = push_from(walked_board, user_pos, move, row_delta, col_delta)
            if pushed is None:
                continue

            pushed_board, push_move = pushed
            yield pushed_board, walk_path + push_move


def apply_single_move(board, move):
    if move not in {direction[0] for direction in DIRECTIONS}:
        raise ValueError(f"Invalid move {move!r}.")

    _, row_delta, col_delta = next(
        direction for direction in DIRECTIONS if direction[0] == move
    )
    user_row, user_col = find_user(board)
    next_row = user_row + row_delta
    next_col = user_col + col_delta
    push_row = user_row + (row_delta * 2)
    push_col = user_col + (col_delta * 2)

    if not in_bounds(board, next_row, next_col):
        return board
    if board[next_row][next_col] == "B":
        return board
    if board[next_row][next_col] in MOVABLE_PIECES:
        if not in_bounds(board, push_row, push_col):
            return board
        if board[push_row][push_col] != "":
            return board

    new_board = [list(row) for row in board]
    if board[next_row][next_col] in MOVABLE_PIECES:
        new_board[push_row][push_col] = board[next_row][next_col]
    new_board[next_row][next_col] = "U"
    new_board[user_row][user_col] = ""
    return tuple(tuple(row) for row in new_board)


def solve(start_board, progress_every=100_000, max_states=None):
    start_board = normalize_board(start_board)
    counter = itertools.count()
    queue = []
    best_cost_seen = {start_board: 0}
    states_checked = 0

    heapq.heappush(queue, (heuristic(start_board), 0, next(counter), start_board, ""))

    while queue:
        _, moves_so_far, _, current_board, moves = heapq.heappop(queue)

        if best_cost_seen.get(current_board, float("inf")) < moves_so_far:
            continue

        states_checked += 1
        if progress_every and states_checked % progress_every == 0:
            print(states_checked, flush=True)

        if pruned_board(current_board):
            if max_states is not None and states_checked >= max_states:
                return None, None, states_checked
            continue

        if solved(current_board):
            return moves, current_board, states_checked

        if max_states is not None and states_checked >= max_states:
            return None, None, states_checked

        for next_board, move_string in next_boards(current_board):
            if pruned_board(next_board):
                continue

            next_cost = moves_so_far + len(move_string)
            if next_cost >= best_cost_seen.get(next_board, float("inf")):
                continue

            best_cost_seen[next_board] = next_cost
            priority = next_cost + heuristic(next_board)
            heapq.heappush(
                queue,
                (priority, next_cost, next(counter), next_board, moves + move_string),
            )

    return None, None, states_checked


def print_board(board):
    for row in board:
        print(" ".join(cell if cell else "." for cell in row))
    print()


def replay_moves(start_board, moves):
    board = start_board
    print("=== STEP BY STEP REPLAY ===")
    print("Start:")
    print_board(board)

    for step, move in enumerate(moves, start=1):
        board = apply_single_move(board, move)
        print("Step", step, "Move:", move)
        print_board(board)


def play_solution(moves):
    import pyautogui

    webbrowser.open_new("https://www.google.com/search?q=tic+tac+go")
    time.sleep(2)

    # Google changes this page occasionally; keep autoplay opt-in from the CLI.
    pyautogui.click(510, 531)
    time.sleep(1.6)

    for move in moves:
        pyautogui.press(KEY_MAP[move])


def parse_board_from_screenshot(screenshot_path, debug_dir=None, no_fallback=False):
    repo_root = Path(__file__).resolve().parents[1]
    parser_dir = repo_root / "boardParsers"
    if str(parser_dir) not in sys.path:
        sys.path.insert(0, str(parser_dir))

    from openCVBoardParser import parse_board_from_image

    parsed_board = parse_board_from_image(
        screenshot_path,
        use_fallback=not no_fallback,
        debug_dir=debug_dir,
    )
    return normalize_board(parsed_board)


def load_start_board(args):
    if args.screenshot:
        return parse_board_from_screenshot(
            args.screenshot,
            debug_dir=args.debug_dir,
            no_fallback=args.no_fallback,
        )

    return normalize_board(DEFAULT_BOARD)


def build_parser():
    parser = argparse.ArgumentParser(description="Solve a Google Tic Tac Go board.")
    parser.add_argument(
        "--screenshot",
        help="Optional path to a saved board screenshot to parse with OpenCV.",
    )
    parser.add_argument(
        "--debug-dir",
        help="Optional directory where the OpenCV board crop is written.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not call the Gemini fallback parser if OpenCV parsing fails.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Open Google Tic Tac Go and press the solved moves with pyautogui.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Do not print periodic state-count progress while searching.",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        help="Stop searching after this many checked states.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    start_board = load_start_board(args)

    print("=== START BOARD ===")
    print_board(start_board)

    start_time = datetime.datetime.now()
    moves, final_board, states_checked = solve(
        start_board,
        progress_every=0 if args.quiet_progress else 100_000,
        max_states=args.max_states,
    )
    end_time = datetime.datetime.now()

    if moves is None:
        if args.max_states is not None and states_checked >= args.max_states:
            print(f"No solution found before --max-states={args.max_states}.")
        else:
            print("Unsolveable")
        print("States checked:", states_checked)
    else:
        replay_moves(start_board, moves)
        print("=== SOLVED ===")
        print("Moves:", moves)
        print("Final Board:")
        print_board(final_board)
        print("States checked:", states_checked)

        if args.autoplay:
            play_solution(moves)

    print("Time Taken:", end_time - start_time)


if __name__ == "__main__":
    main()
