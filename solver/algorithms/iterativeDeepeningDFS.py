"""Iterative deepening depth-first search for Tic Tac Go boards.

This is intentionally separate from beamSearch.py so it can be tested as a
plain search baseline. It expands raw U/D/L/R moves and increases the depth
limit until it finds a solution or reaches max_depth.
"""

from __future__ import annotations

import torch as th
from time import monotonic


ACTION_MOVES = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)

HEURISTIC_WEIGHT = 1.0
MODEL_ACTION_WEIGHT = 1.0
OPEN_BOARD_B_LIMIT = 7
AGENT_O_CLOSE_DISTANCE = 3
AGENT_O_DISTANCE_WEIGHT = 0.4
PATH_BLOCKER_WEIGHT = 0.5


def normalize_board(board):
    return tuple(
        tuple("" if cell == "." else cell for cell in row)
        for row in board
    )


def find_user(board):
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell == "U":
                return row_index, col_index
    return None


def move(board, row_change, col_change):
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


def board_to_rows(board):
    return [" ".join(cell if cell else "." for cell in row) for row in board]


def model_action_scores(model, board):
    if model is None or not hasattr(model, "get_obs"):
        return None

    obs = model.get_obs(board_to_rows(board)).unsqueeze(0)
    try:
        device = next(model.parameters()).device
        obs = obs.to(device)
    except StopIteration:
        pass

    with th.no_grad():
        return model(obs)[0].detach().cpu()


def valid_win_lines(board):
    lines = []
    for row in range(len(board)):
        for col in range(len(board[row]) - 2):
            line = [(row, col), (row, col + 1), (row, col + 2)]
            if all(board[line_row][line_col] != "B" for line_row, line_col in line):
                lines.append(line)

    for row in range(len(board) - 2):
        for col in range(len(board[row])):
            line = [(row, col), (row + 1, col), (row + 2, col)]
            if all(board[line_row][line_col] != "B" for line_row, line_col in line):
                lines.append(line)

    return lines


def line_score(board, line, useful_positions):
    occupied = 0
    blockers = 0
    distance = 0

    for target_row, target_col in line:
        cell = board[target_row][target_col]
        if cell in ("O", "U"):
            occupied += 1
        elif cell == "X":
            blockers += 1

        distance += min(
            abs(piece_row - target_row) + abs(piece_col - target_col)
            for piece_row, piece_col in useful_positions
        )

    return distance - (occupied * 3) + (blockers * 4)


def l_path_cells(start, end, row_first):
    start_row, start_col = start
    end_row, end_col = end
    cells = []

    if row_first:
        col_step = 1 if end_col >= start_col else -1
        for col in range(start_col, end_col + col_step, col_step):
            cells.append((start_row, col))

        row_step = 1 if end_row >= start_row else -1
        for row in range(start_row + row_step, end_row + row_step, row_step):
            cells.append((row, end_col))
    else:
        row_step = 1 if end_row >= start_row else -1
        for row in range(start_row, end_row + row_step, row_step):
            cells.append((row, start_col))

        col_step = 1 if end_col >= start_col else -1
        for col in range(start_col + col_step, end_col + col_step, col_step):
            cells.append((end_row, col))

    return cells[1:-1]


def count_path_blockers(board, agent_position, o_positions):
    blockers = 0
    for o_position in o_positions:
        row_first_blockers = sum(
            board[row][col] == "X"
            for row, col in l_path_cells(agent_position, o_position, True)
        )
        col_first_blockers = sum(
            board[row][col] == "X"
            for row, col in l_path_cells(agent_position, o_position, False)
        )
        blockers += min(row_first_blockers, col_first_blockers)

    return blockers


def o_can_be_pushed_somewhere(board, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(check_row, check_col):
        return 0 <= check_row < len(board) and 0 <= check_col < len(board[check_row])

    def is_wall(check_row, check_col):
        return not in_bounds(check_row, check_col) or board[check_row][check_col] == "B"

    for row_change, col_change in directions:
        push_from_row = row - row_change
        push_from_col = col - col_change
        push_to_row = row + row_change
        push_to_col = col + col_change

        if is_wall(push_from_row, push_from_col):
            continue
        if is_wall(push_to_row, push_to_col):
            continue
        return True

    return False


def line_completion_heuristic(board):
    agent_position = None
    o_positions = []
    useful_positions = [
        (row_index, col_index)
        for row_index, row in enumerate(board)
        for col_index, cell in enumerate(row)
        if cell in ("O", "U")
    ]
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell == "U":
                agent_position = (row_index, col_index)
            elif cell == "O":
                o_positions.append((row_index, col_index))

    if len(useful_positions) < 3:
        return 0.0

    scores = sorted(
        line_score(board, line, useful_positions)
        for line in valid_win_lines(board)
    )
    if not scores:
        return -1_000_000.0

    best_scores = scores[:3]
    heuristic = -float(sum(best_scores) / len(best_scores))

    num_bs = sum(cell == "B" for row in board for cell in row)
    if num_bs <= OPEN_BOARD_B_LIMIT and agent_position is not None and o_positions:
        path_target_os = [
            o_position
            for o_position in o_positions
            if o_can_be_pushed_somewhere(board, o_position[0], o_position[1])
        ] or o_positions
        nearest_o_distance = min(
            abs(agent_position[0] - o_position[0])
            + abs(agent_position[1] - o_position[1])
            for o_position in path_target_os
        )
        heuristic -= AGENT_O_DISTANCE_WEIGHT * max(
            0,
            nearest_o_distance - AGENT_O_CLOSE_DISTANCE,
        )
        heuristic -= PATH_BLOCKER_WEIGHT * count_path_blockers(
            board,
            agent_position,
            path_target_os,
        )

    return heuristic


def solve(
    board,
    model=None,
    max_depth=200,
    depth_step=20,
    timeout_seconds=None,
    action_order="UDLR",
    prune_lost=True,
    model_action_weight=MODEL_ACTION_WEIGHT,
    progress_callback=None,
):
    """Return (moves, final_board, states_checked).

    Empty cells may be either "" or "." in the input. If no solution is found,
    moves is "" and final_board is the normalized start board.
    """
    start_board = normalize_board(board)
    start_time = monotonic()
    states_checked = 0

    if solved(start_board):
        return "", start_board, states_checked

    action_map = {name: (row_change, col_change) for name, row_change, col_change in ACTION_MOVES}
    ordered_actions = [
        (name, *action_map[name])
        for name in action_order
        if name in action_map
    ]

    def timed_out():
        return (
            timeout_seconds is not None
            and monotonic() - start_time >= timeout_seconds
        )

    def ordered_candidates(current_board, path_seen):
        scores = model_action_scores(model, current_board)
        candidates = []

        for action_index, (move_name, row_change, col_change) in enumerate(ordered_actions):
            next_board = move(current_board, row_change, col_change)
            if next_board == current_board:
                continue
            if next_board in path_seen:
                continue
            if prune_lost and lost_check(next_board):
                continue

            heuristic = line_completion_heuristic(next_board)
            model_score = float(scores[action_index]) if scores is not None else 0.0
            score = (HEURISTIC_WEIGHT * heuristic) + (
                model_action_weight * model_score
            )
            candidates.append((score, move_name, next_board))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def depth_limited_search(current_board, depth_remaining, moves, path_seen):
        nonlocal states_checked
        states_checked += 1

        if timed_out():
            return None
        if solved(current_board):
            return moves, current_board
        if depth_remaining == 0:
            return None
        if prune_lost and lost_check(current_board):
            return None

        for _score, move_name, next_board in ordered_candidates(current_board, path_seen):
            path_seen.add(next_board)
            result = depth_limited_search(
                next_board,
                depth_remaining - 1,
                moves + move_name,
                path_seen,
            )
            path_seen.remove(next_board)

            if result is not None:
                return result

        return None

    for depth_limit in range(depth_step, max_depth + 1, depth_step):
        if timed_out():
            break

        states_before_depth = states_checked
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "depth_start",
                    "depth_limit": depth_limit,
                    "states_checked": states_checked,
                    "elapsed_seconds": monotonic() - start_time,
                }
            )

        result = depth_limited_search(start_board, depth_limit, "", {start_board})
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "depth_end",
                    "depth_limit": depth_limit,
                    "states_checked": states_checked,
                    "states_this_depth": states_checked - states_before_depth,
                    "elapsed_seconds": monotonic() - start_time,
                    "solved": result is not None,
                }
            )
        if result is not None:
            moves, final_board = result
            return moves, final_board, states_checked

    return "", start_board, states_checked


if __name__ == "__main__":
    example = (
        ("", "", ""),
        ("U", "O", ""),
        ("", "O", ""),
    )
    moves, final_board, checked = solve(example, max_depth=20, depth_step=20)
    print(f"moves={moves!r} states_checked={checked}")
    for row in board_to_rows(final_board):
        print(row)
