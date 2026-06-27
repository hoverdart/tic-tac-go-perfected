"""Diagnose how the beam heuristic ranks known solution moves.

Fill KNOWN_SOLUTIONS with entries like:

    (16, 37): "RRULDD"

The key is (grad, zero_based_board_index). Board number 38 is index 37.
"""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_REGISTER_DIR = REPO_ROOT / "solver" / "gymnasium_register"
TRAINING_BOARDS_PATH = GYM_REGISTER_DIR / "generated_training_boards.py"
REPORT_PATH = GYM_REGISTER_DIR / "known_solution_heuristic_diagnostic.csv"

HEURISTIC_WEIGHT = 1.0
OPEN_BOARD_B_LIMIT = 7
AGENT_O_CLOSE_DISTANCE = 3
AGENT_O_DISTANCE_WEIGHT = 0.4
PATH_BLOCKER_WEIGHT = 0.5

ACTION_MOVES = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)

# Add known solutions here.
KNOWN_SOLUTIONS = {
    # (16, 37): "PUT_MOVES_HERE",
    (17, 62): "UURDLDRRRURLLLULUURDDLDRRRDRU",
    (17, 106): "UUUDRDRRUULLRRRRUULDRDLLLDL",
    (17, 107): "URULUULLLLLDLDRDRDLUUUURRRUULDRDLLULDDDD",
    (17, 176): "DLDLLULULLDRRURDDRDRDDDLLDLUULURLLLURUURRRD",
}

FIELDNAMES = [
    "grad",
    "board_index",
    "board_number",
    "board_line",
    "step",
    "correct_move",
    "correct_rank",
    "candidate_count",
    "correct_score",
    "best_move",
    "best_score",
    "score_gap",
    "correct_in_top_2",
    "correct_in_top_4",
    "correct_in_top_10",
    "correct_was_legal",
    "correct_was_pruned_softlock",
    "correct_was_loss",
    "solved_after_step",
]

if str(GYM_REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_REGISTER_DIR))

from generated_training_boards import TRAINING_BOARDS  # noqa: E402


def training_board_line_numbers(grads):
    tree = ast.parse(TRAINING_BOARDS_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "TRAINING_BOARDS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        line_numbers = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if not isinstance(key_node, ast.Constant):
                continue
            grad = key_node.value
            if grad not in grads or not isinstance(value_node, ast.List):
                continue
            line_numbers[grad] = [board_node.lineno for board_node in value_node.elts]
        return line_numbers

    raise ValueError(f"Could not find TRAINING_BOARDS in {TRAINING_BOARDS_PATH}")


def board_to_text(board):
    return "\n".join(" ".join(cell if cell else "." for cell in row) for row in board)


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


def obvious_soft_locked(board):
    def in_bounds(row, col):
        return 0 <= row < len(board) and 0 <= col < len(board[row])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_wall(row, col):
        return not in_bounds(row, col) or board[row][col] == "B"

    def is_surrounded_by_walls(row, col):
        return all(is_wall(row + dr, col + dc) for dr, dc in directions)

    def o_can_be_pushed_somewhere_local(row, col):
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

    useful_positions = [
        (row, col)
        for row in range(len(board))
        for col in range(len(board[row]))
        if board[row][col] != "B"
    ]
    if not useful_positions:
        return True

    min_row = min(row for row, _ in useful_positions)
    max_row = max(row for row, _ in useful_positions)
    min_col = min(col for _, col in useful_positions)
    max_col = max(col for _, col in useful_positions)

    def on_playable_edge(row, col):
        return row in (min_row, max_row) or col in (min_col, max_col)

    def has_near_line_slot(o_locations):
        if len(o_locations) != 2:
            return False

        first_o, second_o = sorted(o_locations)
        row_distance = abs(first_o[0] - second_o[0])
        col_distance = abs(first_o[1] - second_o[1])

        if first_o[0] == second_o[0] and col_distance == 1:
            row = first_o[0]
            left_col = min(first_o[1], second_o[1]) - 1
            right_col = max(first_o[1], second_o[1]) + 1
            return (
                in_bounds(row, left_col)
                and board[row][left_col] != "B"
            ) or (
                in_bounds(row, right_col)
                and board[row][right_col] != "B"
            )

        if first_o[0] == second_o[0] and col_distance == 2:
            row = first_o[0]
            middle_col = (first_o[1] + second_o[1]) // 2
            return board[row][middle_col] != "B"

        if first_o[1] == second_o[1] and row_distance == 1:
            col = first_o[1]
            top_row = min(first_o[0], second_o[0]) - 1
            bottom_row = max(first_o[0], second_o[0]) + 1
            return (
                in_bounds(top_row, col)
                and board[top_row][col] != "B"
            ) or (
                in_bounds(bottom_row, col)
                and board[bottom_row][col] != "B"
            )

        if first_o[1] == second_o[1] and row_distance == 2:
            middle_row = (first_o[0] + second_o[0]) // 2
            col = first_o[1]
            return board[middle_row][col] != "B"

        return False

    o_locations = []
    user_location = None
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == "O":
                o_locations.append((row, col))
            elif board[row][col] == "U":
                user_location = (row, col)

    if user_location is None:
        return True

    if is_surrounded_by_walls(*user_location):
        return True

    if any(is_surrounded_by_walls(row, col) for row, col in o_locations):
        return True

    if len(o_locations) == 2:
        near_line_slot = has_near_line_slot(o_locations)
        if (
            not near_line_slot
            and all(on_playable_edge(row, col) for row, col in o_locations)
            and not any(
                o_can_be_pushed_somewhere_local(row, col)
                for row, col in o_locations
            )
        ):
            return True

        if (
            not near_line_slot
            and not any(
                o_can_be_pushed_somewhere_local(row, col)
                for row, col in o_locations
            )
        ):
            return True

    return False


def legal_candidates(board):
    candidates = []
    for move_name, row_change, col_change in ACTION_MOVES:
        next_board = move(board, row_change, col_change)
        if next_board == board:
            continue

        is_loss = lost_check(next_board)
        is_softlock = not solved(next_board) and obvious_soft_locked(next_board)
        if is_loss or is_softlock:
            continue

        score = HEURISTIC_WEIGHT * line_completion_heuristic(next_board)
        candidates.append((score, move_name, next_board))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return candidates


def diagnose_solution(grad, board_index, solution, board_line):
    board = tuple(tuple(row) for row in TRAINING_BOARDS[grad][board_index])
    rows = []

    print()
    print(f"Grad {grad} board {board_index + 1} line {board_line}")
    print("=== START BOARD ===")
    print(board_to_text(board))

    for step_index, correct_move in enumerate(solution, start=1):
        candidates = legal_candidates(board)
        move_rows = [
            (rank, score, move_name, next_board)
            for rank, (score, move_name, next_board) in enumerate(candidates, start=1)
        ]
        correct = next(
            (
                (rank, score, next_board)
                for rank, score, move_name, next_board in move_rows
                if move_name == correct_move
            ),
            None,
        )

        correct_next_board = move(
            board,
            next(
                (row_change, col_change)
                for move_name, row_change, col_change in ACTION_MOVES
                if move_name == correct_move
            )[0],
            next(
                (row_change, col_change)
                for move_name, row_change, col_change in ACTION_MOVES
                if move_name == correct_move
            )[1],
        )
        correct_was_legal = correct_next_board != board
        correct_was_loss = correct_was_legal and lost_check(correct_next_board)
        correct_was_pruned_softlock = (
            correct_was_legal
            and not solved(correct_next_board)
            and obvious_soft_locked(correct_next_board)
        )

        best_move = candidates[0][1] if candidates else ""
        best_score = candidates[0][0] if candidates else ""
        if correct is None:
            correct_rank = ""
            correct_score = ""
            score_gap = ""
            next_board = correct_next_board
        else:
            correct_rank, correct_score, next_board = correct
            score_gap = float(best_score) - correct_score if candidates else ""

        solved_after_step = solved(next_board)
        row = {
            "grad": grad,
            "board_index": board_index,
            "board_number": board_index + 1,
            "board_line": board_line,
            "step": step_index,
            "correct_move": correct_move,
            "correct_rank": correct_rank,
            "candidate_count": len(candidates),
            "correct_score": correct_score,
            "best_move": best_move,
            "best_score": best_score,
            "score_gap": score_gap,
            "correct_in_top_2": bool(correct and correct_rank <= 2),
            "correct_in_top_4": bool(correct and correct_rank <= 4),
            "correct_in_top_10": bool(correct and correct_rank <= 10),
            "correct_was_legal": correct_was_legal,
            "correct_was_pruned_softlock": correct_was_pruned_softlock,
            "correct_was_loss": correct_was_loss,
            "solved_after_step": solved_after_step,
        }
        rows.append(row)

        rank_text = correct_rank if correct_rank != "" else "PRUNED"
        print(
            f"step {step_index:03d} correct={correct_move} rank={rank_text} "
            f"candidates={len(candidates)} best={best_move} "
            f"gap={score_gap if score_gap != '' else 'n/a'}"
        )

        board = next_board
        if solved_after_step:
            break

    bad_rows = [
        row for row in rows
        if row["correct_rank"] == "" or int(row["correct_rank"]) > 4
    ]
    if bad_rows:
        first_bad = bad_rows[0]
        print(
            "First weak step: "
            f"{first_bad['step']} rank={first_bad['correct_rank'] or 'PRUNED'}"
        )
    else:
        print("All known moves ranked top 4 locally.")

    return rows


def main():
    if not KNOWN_SOLUTIONS:
        raise SystemExit(
            "Add entries to KNOWN_SOLUTIONS first, e.g. "
            "{(16, 37): 'RRULDD'}."
        )

    grads = sorted({grad for grad, _ in KNOWN_SOLUTIONS})
    line_numbers = training_board_line_numbers(grads)
    rows = []

    for (grad, board_index), solution in sorted(KNOWN_SOLUTIONS.items()):
        rows.extend(
            diagnose_solution(
                grad,
                board_index,
                solution,
                line_numbers[grad][board_index],
            )
        )

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote diagnostic CSV: {REPORT_PATH}")


if __name__ == "__main__":
    main()
