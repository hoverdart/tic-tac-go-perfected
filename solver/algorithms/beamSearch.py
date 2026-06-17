import torch as th
import numpy as np
from stable_baselines3 import DQN
from collections import deque

def beamSearch(initial_board, model, beam_width, max_depth):
    def lostCheck(self, board):
            for i in range(0, len(board)):
                for j in range(0, len(board[i]) - 2):
                    if board[i][j] == "X" and board[i][j + 1] == "X" and board[i][j + 2] == "X":
                        return True
    
            for i in range(0, len(board) - 2):
                for j in range(0, len(board[i])):
                    if board[i][j] == "X" and board[i + 1][j] == "X" and board[i + 2][j] == "X":
                        return True
    
            return False
    
    def solved(self, board):
        for i in range(0, len(board)):
            for j in range(0, len(board[i]) - 2):
                if board[i][j] in ("O", "U") and board[i][j + 1] in ("O", "U") and board[i][j + 2] in ("O", "U"):
                    return True

        for i in range(0, len(board) - 2):
            for j in range(0, len(board[i])):
                if board[i][j] in ("O", "U") and board[i + 1][j] in ("O", "U") and board[i + 2][j] in ("O", "U"):
                    return True

        return False

    def find_user(self, board):
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == "U":
                    return i, j

        return None

    def move(self, board, row_change, col_change):
        userPos = self.find_user(board)
        if userPos is None:
            return board

        user_row, user_col = userPos
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

        newBoard = [list(row) for row in board]
        if next_cell in ("X", "O"):
            newBoard[push_row][push_col] = next_cell

        newBoard[next_row][next_col] = "U"
        newBoard[user_row][user_col] = ""
        return tuple(tuple(row) for row in newBoard)

    def moveUp(self, board):
        return self.move(board, -1, 0)

    def moveDown(self, board):
        return self.move(board, 1, 0)

    def moveLeft(self, board):
        return self.move(board, 0, -1)

    def moveRight(self, board):
        return self.move(board, 0, 1)

    def board_config_key(board):
        return tuple(
            tuple("" if cell == "U" else cell for cell in row)
            for row in board
        )

    def user_position_for_board(board):
        for row_index, row in enumerate(board):
            for col_index, cell in enumerate(row):
                if cell == "U":
                    return row_index, col_index
        return None

    def remember_agent_position_for_config(board, visited_config_positions):
        user_position = user_position_for_board(board)
        if user_position is None:
            return
        config_key = board_config_key(board)
        visited_config_positions.setdefault(config_key, set()).add(user_position)

    def copy_visited_config_positions(visited_config_positions):
        return {
            config_key: set(positions)
            for config_key, positions in visited_config_positions.items()
        }

    def get_obs(board, visited_config_positions=None):
        mapping = {"":0, "X":1, "O":2, "U":3, "B":4}
        arr = [[mapping[cell] for cell in row] for row in board]

        compatibleBoard = np.array(arr, dtype=np.int32)

        threeDArr = np.zeros((6, 8, 8), dtype=np.int32)

        for i in range(0, len(compatibleBoard)):
                for j in range(0, len(compatibleBoard[i])):
                    if compatibleBoard[i][j] == 0:
                        threeDArr[0][i][j] = 1
                    elif compatibleBoard[i][j] == 1:
                        threeDArr[1][i][j] = 1
                    elif compatibleBoard[i][j] == 2:
                        threeDArr[2][i][j] = 1
                    elif compatibleBoard[i][j] == 3:
                        threeDArr[3][i][j] = 1
                    elif compatibleBoard[i][j] == 4:
                        threeDArr[4][i][j] = 1

        if visited_config_positions is not None:
            config_key = board_config_key(board)
            for row, col in visited_config_positions.get(config_key, set()):
                threeDArr[5][row][col] = 1

        return threeDArr

    def softLocked(self, board):
        def in_bounds(row, col):
            return 0 <= row < len(board) and 0 <= col < len(board[row])

        def x_is_movable(row, col):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for row_change, col_change in directions:
                push_from_row = row - row_change
                push_from_col = col - col_change
                push_to_row = row + row_change
                push_to_col = col + col_change

                if not in_bounds(push_from_row, push_from_col):
                    continue
                if not in_bounds(push_to_row, push_to_col):
                    continue
                if board[push_from_row][push_from_col] not in ("", "U"):
                    continue
                if board[push_to_row][push_to_col] == "":
                    return True

            return False

        def spot_can_become_user(row, col):
            if board[row][col] in ("", "U"):
                return True
            if board[row][col] == "X":
                return x_is_movable(row, col)
            return False

        def spot_can_become_empty(row, col):
            if board[row][col] == "":
                return True
            if board[row][col] == "X":
                return x_is_movable(row, col)
            return False

        def o_is_movable(row, col):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for row_change, col_change in directions:
                push_from_row = row - row_change
                push_from_col = col - col_change
                push_to_row = row + row_change
                push_to_col = col + col_change

                if not in_bounds(push_from_row, push_from_col):
                    continue
                if not in_bounds(push_to_row, push_to_col):
                    continue
                if not spot_can_become_user(push_from_row, push_from_col):
                    continue
                if spot_can_become_empty(push_to_row, push_to_col):
                    return True

            return False

        def o_can_move_direction(row, col, row_change, col_change):
            push_from_row = row - row_change
            push_from_col = col - col_change
            push_to_row = row + row_change
            push_to_col = col + col_change

            if not in_bounds(push_from_row, push_from_col):
                return False
            if not in_bounds(push_to_row, push_to_col):
                return False
            if not spot_can_become_user(push_from_row, push_from_col):
                return False
            return spot_can_become_empty(push_to_row, push_to_col)

        o_locations = []
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == "O":
                    o_locations.append((i, j))

        if len(o_locations) == 2:
            first_o, second_o = o_locations
            os_are_aligned = first_o[0] == second_o[0] or first_o[1] == second_o[1]
            if not os_are_aligned:
                first_o_movable = o_is_movable(first_o[0], first_o[1])
                second_o_movable = o_is_movable(second_o[0], second_o[1])
                if not first_o_movable and not second_o_movable:
                    return True

                left_o, right_o = sorted(o_locations, key=lambda location: location[1])
                if right_o[1] - left_o[1] > 2:
                    left_can_move_right = o_can_move_direction(left_o[0], left_o[1], 0, 1)
                    right_can_move_left = o_can_move_direction(right_o[0], right_o[1], 0, -1)
                    left_can_move_vertically = (
                        o_can_move_direction(left_o[0], left_o[1], -1, 0)
                        or o_can_move_direction(left_o[0], left_o[1], 1, 0)
                    )
                    right_can_move_vertically = (
                        o_can_move_direction(right_o[0], right_o[1], -1, 0)
                        or o_can_move_direction(right_o[0], right_o[1], 1, 0)
                    )
                    if (
                        not left_can_move_right
                        and not right_can_move_left
                        and not left_can_move_vertically
                        and not right_can_move_vertically
                    ):
                        return True

                top_o, bottom_o = sorted(o_locations, key=lambda location: location[0])
                if bottom_o[0] - top_o[0] > 2:
                    top_can_move_down = o_can_move_direction(top_o[0], top_o[1], 1, 0)
                    bottom_can_move_up = o_can_move_direction(bottom_o[0], bottom_o[1], -1, 0)
                    top_can_move_sideways = (
                        o_can_move_direction(top_o[0], top_o[1], 0, -1)
                        or o_can_move_direction(top_o[0], top_o[1], 0, 1)
                    )
                    bottom_can_move_sideways = (
                        o_can_move_direction(bottom_o[0], bottom_o[1], 0, -1)
                        or o_can_move_direction(bottom_o[0], bottom_o[1], 0, 1)
                    )
                    if (
                        not top_can_move_down
                        and not bottom_can_move_up
                        and not top_can_move_sideways
                        and not bottom_can_move_sideways
                    ):
                        return True

        found_two_os_in_line = False

        for i in range(0, len(board)):
            for j in range(0, len(board[i]) - 2):
                line = [board[i][j], board[i][j + 1], board[i][j + 2]]
                if line.count("O") == 2:
                    found_two_os_in_line = True
                    for col in range(j, j + 3):
                        if board[i][col] != "O" and spot_can_become_user(i, col):
                            return False

        for i in range(0, len(board) - 2):
            for j in range(0, len(board[i])):
                line = [board[i][j], board[i + 1][j], board[i + 2][j]]
                if line.count("O") == 2:
                    found_two_os_in_line = True
                    for row in range(i, i + 3):
                        if board[row][j] != "O" and spot_can_become_user(row, j):
                            return False

        return found_two_os_in_line

    def active_size(self, board):
        length = 0
        width = 0

        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != "B":
                    if i > length - 1:
                        length = i + 1
                    if j > width - 1:
                        width = j + 1

        if length == 0:
            length = len(board)
        if width == 0:
            width = len(board[0])

        return length, width

    def replayMoves(
        self,
        startBoard,
        moves,
        current_grad=14,
        terminate_on_repeated_states=False,
        repeat_termination_limit=0,
        penalize_repeated_states=False,
    ):
        board = tuple(tuple(row) for row in startBoard)
        data = []
        visited_states = {board: 1}
        visited_config_positions = {}
        remember_agent_position_for_config(board, visited_config_positions)
        length, width = self.active_size(board)

        for i, move in enumerate(moves):
            current_pos = self.find_user(board)
            state_before_move = self.get_obs(board, visited_config_positions)

            if move == "U":
                action_chose = 0
                next_board = self.moveUp(board)
            elif move == "D":
                action_chose = 1
                next_board = self.moveDown(board)
            elif move == "L":
                action_chose = 2
                next_board = self.moveLeft(board)
            else:
                action_chose = 3
                next_board = self.moveRight(board)

            next_pos = self.find_user(next_board)
            same = current_pos == next_pos

            board_state_count = visited_states.get(next_board, 0) + 1
            visited_states[next_board] = board_state_count
            repeated_too_much = (
                terminate_on_repeated_states
                and current_grad >= 4
                and board_state_count >= repeat_termination_limit
            )

            won = self.solved(next_board)
            lost = self.lostCheck(next_board)
            softLocked = self.softLocked(next_board)

            done = won or lost
            reward_got = -0.8 * (16 / (length * width))

            if softLocked:
                done = True
                reward_got += -10

            if same:
                reward_got += -1

            if repeated_too_much:
                done = True
                if penalize_repeated_states:
                    reward_got += -5

            if lost:
                reward_got += -10
            elif won:
                reward_got += 40

            remember_agent_position_for_config(next_board, visited_config_positions)

            data.append(
                {
                    "observation": state_before_move,
                    "action": action_chose,
                    "reward": reward_got,
                    "next_observation": self.get_obs(
                        next_board,
                        visited_config_positions,
                    ),
                    "done": done,
                }
            )

            board = next_board
            if done:
                break

        return data

    class Helpers:
        pass

    helpers = Helpers()
    helpers.lostCheck = lambda board: lostCheck(helpers, board)
    helpers.solved = lambda board: solved(helpers, board)
    helpers.find_user = lambda board: find_user(helpers, board)
    helpers.move = lambda board, row_change, col_change: move(
        helpers, board, row_change, col_change
    )
    helpers.moveUp = lambda board: moveUp(helpers, board)
    helpers.moveDown = lambda board: moveDown(helpers, board)
    helpers.moveLeft = lambda board: moveLeft(helpers, board)
    helpers.moveRight = lambda board: moveRight(helpers, board)
    helpers.get_obs = get_obs
    helpers.softLocked = lambda board: softLocked(helpers, board)
    helpers.active_size = lambda board: active_size(helpers, board)
    helpers.replayMoves = lambda *args, **kwargs: replayMoves(
        helpers, *args, **kwargs
    )

    def solve(
        board,
        preMoves=None,
        current_grad=14,
        terminate_on_repeated_states=False,
    ):
        start_board = tuple(tuple(row) for row in board)
        currentBoards = deque()
        visited = set()
        start_config_positions = {}
        remember_agent_position_for_config(start_board, start_config_positions)
        currentBoards.append((start_board, "", 0, start_config_positions))
        visited.add(start_board)
        statesChecked = 0
        action_moves = [
            (0, "U", helpers.moveUp),
            (1, "D", helpers.moveDown),
            (2, "L", helpers.moveLeft),
            (3, "R", helpers.moveRight),
        ]
    
        while currentBoards:
            depth_size = len(currentBoards)
            candidates = []

            for _ in range(depth_size):
                currentBoard, moves, depth, visited_config_positions = currentBoards.popleft()
                statesChecked += 1

                if statesChecked % 100000 == 0:
                    print(statesChecked)

                if helpers.lostCheck(currentBoard):
                    continue

                if helpers.solved(currentBoard):
                    print("=== SOLVED ===")
                    print("Moves:", moves)
                    print("States checked: ", statesChecked)
                    return (
                        moves,
                        helpers.replayMoves(
                            start_board,
                            moves,
                            current_grad=current_grad,
                            terminate_on_repeated_states=terminate_on_repeated_states,
                        ),
                    )

                if helpers.softLocked(currentBoard):
                    continue

                if depth >= max_depth:
                    continue

                obs = helpers.get_obs(currentBoard, visited_config_positions)
                obs_tensor = th.tensor(
                    obs, dtype=th.float32, device=model.device
                ).unsqueeze(0)

                with th.no_grad():
                    q_values = model.q_net(obs_tensor).cpu().numpy().reshape(-1)

                accepted_from_state = 0
                for action_index in np.argsort(q_values)[::-1]:
                    _, move_name, move_fn = action_moves[int(action_index)]
                    nextBoard = move_fn(currentBoard)

                    # Illegal moves leave the board unchanged.
                    if nextBoard == currentBoard:
                        continue
                    if nextBoard in visited:
                        continue
                    if helpers.lostCheck(nextBoard):
                        continue
                    if not helpers.solved(nextBoard) and helpers.softLocked(nextBoard):
                        continue

                    next_visited_config_positions = copy_visited_config_positions(
                        visited_config_positions
                    )
                    remember_agent_position_for_config(
                        nextBoard,
                        next_visited_config_positions,
                    )

                    candidates.append(
                        (
                            float(q_values[int(action_index)]),
                            nextBoard,
                            moves + move_name,
                            depth + 1,
                            next_visited_config_positions,
                        )
                    )
                    visited.add(nextBoard)
                    accepted_from_state += 1

                    if accepted_from_state == 2:
                        break

            candidates.sort(key=lambda candidate: candidate[0], reverse=True)
            currentBoards.extend(
                (board, moves, depth, visited_config_positions)
                for _, board, moves, depth, visited_config_positions
                in candidates[:beam_width]
            )
    
        print("No solution found. States checked:", statesChecked)
        return "", []


    return solve(initial_board)
