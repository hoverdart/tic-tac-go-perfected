import numpy as np
import heapq
import itertools
import logging
import random
import time
import torch as th
from collections import deque

logger = logging.getLogger(__name__)

def beamSearch(
    initial_board,
    model,
    beam_width,
    max_depth,
    debug=False,
    random_tiebreak=False,
    seed=None,
    tiebreak_noise=1e-6,
    restarts=1,
    random_prefix_steps=None,
    model_action_weight=1,
    restart_model_action_weights=None,
    timeout_seconds=None,
):
    """Run beam search with optional CNN action scores and randomized restarts."""
    HEURISTIC_WEIGHT = 1.0
    OPEN_BOARD_B_LIMIT = 7
    AGENT_O_CLOSE_DISTANCE = 3
    AGENT_O_DISTANCE_WEIGHT = 0.4
    PATH_BLOCKER_WEIGHT = 0.5
    deadline = (
        time.perf_counter() + timeout_seconds
        if timeout_seconds is not None
        else None
    )

    def timed_out():
        """Return True once the optional solve deadline has elapsed."""
        return deadline is not None and time.perf_counter() >= deadline

    def lostCheck(self, board):
            """Return True if any three X pieces form a horizontal/vertical line."""
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
        """Return True if O/O/U occupy any horizontal or vertical line of 3."""
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
        """Find the U piece position, or None if the board has no U."""
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == "U":
                    return i, j

        return None

    def move(self, board, row_change, col_change):
        """Move U one cell, optionally pushing one adjacent X/O into empty space."""
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
        """Move U up one cell if legal."""
        return self.move(board, -1, 0)

    def moveDown(self, board):
        """Move U down one cell if legal."""
        return self.move(board, 1, 0)

    def moveLeft(self, board):
        """Move U left one cell if legal."""
        return self.move(board, 0, -1)

    def moveRight(self, board):
        """Move U right one cell if legal."""
        return self.move(board, 0, 1)

    def board_config_key(board):
        """Hash board layout without U position, for repeated-position memory."""
        return tuple(
            tuple("" if cell == "U" else cell for cell in row)
            for row in board
        )

    def user_position_for_board(board):
        """Return the U position for repeat-memory bookkeeping."""
        for row_index, row in enumerate(board):
            for col_index, cell in enumerate(row):
                if cell == "U":
                    return row_index, col_index
        return None

    def remember_agent_position_for_config(board, visited_config_positions):
        """Record where U has stood for the current non-U board layout."""
        user_position = user_position_for_board(board)
        if user_position is None:
            return
        config_key = board_config_key(board)
        visited_config_positions.setdefault(config_key, set()).add(user_position)

    def copy_visited_config_positions(visited_config_positions):
        """Deep-copy layout-to-agent-position history for a branch."""
        return {
            config_key: set(positions)
            for config_key, positions in visited_config_positions.items()
        }

    def get_obs(board, visited_config_positions=None):
        """Build the 6-channel DQN-style observation tensor for replay data."""
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

    def board_to_rows(board):
        """Convert tuple board rows into the text rows expected by SmallCNN."""
        return [" ".join(cell if cell else "." for cell in row) for row in board]

    def model_action_scores(board):
        """Return CNN logits for U/D/L/R, or None when no compatible model exists."""
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
        """List all non-barrier horizontal/vertical 3-cell target lines."""
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
        """Score one target line; lower means easier to complete."""
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
        """Return cells on an L-shaped path between two positions."""
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
        """Count X blockers on the clearer L-path from U to each target O."""
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
        """Return True if an O has at least one non-wall push direction."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def in_bounds(check_row, check_col):
            return (
                0 <= check_row < len(board)
                and 0 <= check_col < len(board[check_row])
            )

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
        """Score board promise from best target lines plus open-board penalties."""
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
        if (
            num_bs <= OPEN_BOARD_B_LIMIT
            and agent_position is not None
            and o_positions
        ):
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

    def softLocked(self, board):
        """Cheaply reject obvious wall/barrier traps that cannot produce a win."""
        def in_bounds(row, col):
            return 0 <= row < len(board) and 0 <= col < len(board[row])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def is_wall(row, col):
            return not in_bounds(row, col) or board[row][col] == "B"

        def is_surrounded_by_walls(row, col):
            return all(is_wall(row + dr, col + dc) for dr, dc in directions)

        def o_can_be_pushed_somewhere(row, col):
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
                and not any(o_can_be_pushed_somewhere(row, col) for row, col in o_locations)
            ):
                return True

            if (
                not near_line_slot
                and not any(o_can_be_pushed_somewhere(row, col) for row, col in o_locations)
            ):
                return True

        return False

    def active_size(self, board):
        """Return active non-barrier board extent used for replay reward scaling."""
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
        """Replay a move string into transition dictionaries for compatibility."""
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

            # if softLocked:
            #     done = True
            #     reward_got += -10

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
        """Small namespace to attach helper methods used by legacy-style code."""
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
        restart_seed=None,
        restart_index=0,
        current_model_action_weight=None,
        original_start_board=None,
        allow_random_prefix=True,
    ):
        """Run one restart of beam search from an optional random prefix."""
        if current_model_action_weight is None:
            current_model_action_weight = model_action_weight

        search_start_board = tuple(tuple(row) for row in board)
        start_board = (
            tuple(tuple(row) for row in original_start_board)
            if original_start_board is not None
            else search_start_board
        )
        initial_moves = preMoves or ""
        initial_depth = len(initial_moves)
        currentBoards = deque()
        best_depth_seen = {search_start_board: initial_depth}
        backup_frontier = []
        backup_counter = itertools.count()
        backup_limit = max(beam_width * 10, beam_width)
        start_config_positions = {}
        remember_agent_position_for_config(search_start_board, start_config_positions)
        currentBoards.append(
            (
                search_start_board,
                initial_moves,
                initial_depth,
                start_config_positions,
            )
        )
        statesChecked = 0
        action_moves = [
            (0, "U", helpers.moveUp),
            (1, "D", helpers.moveDown),
            (2, "L", helpers.moveLeft),
            (3, "R", helpers.moveRight),
        ]
        rng = random.Random(restart_seed)

        def ranked_score(item):
            """Apply optional random noise to near-tie candidate ordering."""
            if not random_tiebreak:
                return item[0]
            return item[0] + rng.uniform(-tiebreak_noise, tiebreak_noise)

        def add_to_backup(candidate):
            """Keep high-scoring pruned candidates for later frontier refills."""
            score = candidate[0]
            entry = (score, next(backup_counter), candidate)
            if len(backup_frontier) < backup_limit:
                heapq.heappush(backup_frontier, entry)
            elif score > backup_frontier[0][0]:
                heapq.heapreplace(backup_frontier, entry)

        def refill_from_backup():
            """Refill the active frontier from the best previously pruned states."""
            refilled = 0
            refill_count = min(beam_width, len(backup_frontier))
            refill_entries = heapq.nlargest(refill_count, backup_frontier)
            refill_ids = {
                counter
                for _score, counter, _candidate in refill_entries
            }
            backup_frontier[:] = [
                entry for entry in backup_frontier
                if entry[1] not in refill_ids
            ]
            heapq.heapify(backup_frontier)

            for _score, _counter, candidate in refill_entries:
                (
                    _score,
                    _q_value,
                    _heuristic,
                    backup_board,
                    backup_moves,
                    backup_depth,
                    backup_visited_config_positions,
                ) = candidate

                if best_depth_seen.get(backup_board) != backup_depth:
                    continue

                currentBoards.append(
                    (
                        backup_board,
                        backup_moves,
                        backup_depth,
                        backup_visited_config_positions,
                    )
                )
                refilled += 1

            return refilled

        def random_prefix(prefix_board, max_prefix_steps):
            """Take a short random legal prefix before starting guided search."""
            prefix_board = tuple(tuple(row) for row in prefix_board)
            prefix_moves = ""
            seen_prefix_boards = {prefix_board}

            for _ in range(max_prefix_steps):
                legal_prefix_moves = []
                for _action_index, move_name, move_fn in action_moves:
                    next_board = move_fn(prefix_board)
                    if next_board == prefix_board:
                        continue
                    if next_board in seen_prefix_boards:
                        continue
                    if helpers.lostCheck(next_board):
                        continue
                    if not helpers.solved(next_board) and helpers.softLocked(next_board):
                        continue
                    legal_prefix_moves.append((move_name, next_board))

                if not legal_prefix_moves:
                    break

                push_moves = [
                    (move_name, next_board)
                    for move_name, next_board in legal_prefix_moves
                    if sum(cell in ("O", "X") for row in next_board for cell in row)
                    == sum(cell in ("O", "X") for row in prefix_board for cell in row)
                    and next_board != prefix_board
                ]
                move_name, prefix_board = rng.choice(push_moves or legal_prefix_moves)
                prefix_moves += move_name
                seen_prefix_boards.add(prefix_board)

                if helpers.solved(prefix_board):
                    break

            return prefix_board, prefix_moves

        prefix_step_options = random_prefix_steps or [0]
        max_prefix_steps = rng.choice(prefix_step_options)
        if allow_random_prefix and max_prefix_steps > 0:
            prefixed_board, prefix_moves = random_prefix(start_board, max_prefix_steps)
            if debug:
                logger.debug(
                    "Restart %s: random_prefix_steps=%s, actual_prefix_moves=%s",
                    restart_index,
                    max_prefix_steps,
                    len(prefix_moves),
                )
            return solve(
                prefixed_board,
                preMoves=prefix_moves,
                current_grad=current_grad,
                terminate_on_repeated_states=terminate_on_repeated_states,
                restart_seed=restart_seed,
                restart_index=restart_index,
                current_model_action_weight=current_model_action_weight,
                original_start_board=start_board,
                allow_random_prefix=False,
            )
    
        while currentBoards:
            if timed_out():
                logger.info(
                    "beam_search.timeout restart=%s states_checked=%s",
                    restart_index,
                    statesChecked,
                )
                return "", []

            depth_size = len(currentBoards)
            candidates = []
            depth = currentBoards[0][2]

            for _ in range(depth_size):
                if timed_out():
                    logger.info(
                        "beam_search.timeout restart=%s states_checked=%s",
                        restart_index,
                        statesChecked,
                    )
                    return "", []

                currentBoard, moves, current_depth, visited_config_positions = currentBoards.popleft()
                statesChecked += 1

                if debug and statesChecked % 100000 == 0:
                    logger.debug("beam_search.states_checked=%s", statesChecked)

                if helpers.lostCheck(currentBoard):
                    continue

                if helpers.solved(currentBoard):
                    logger.info(
                        "beam_search.solved moves=%s states_checked=%s",
                        moves,
                        statesChecked,
                    )
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

                if current_depth >= max_depth:
                    continue

                model_scores = model_action_scores(currentBoard)

                scored_actions = []
                for action_index, move_name, move_fn in action_moves:
                    q_value = (
                        float(model_scores[action_index])
                        if model_scores is not None
                        else 0.0
                    )
                    nextBoard = move_fn(currentBoard)

                    # Illegal moves leave the board unchanged.
                    if nextBoard == currentBoard:
                        continue
                    next_depth = current_depth + 1
                    if next_depth >= best_depth_seen.get(nextBoard, 1_000_000_000):
                        continue
                    if helpers.lostCheck(nextBoard):
                        continue
                    if not helpers.solved(nextBoard) and helpers.softLocked(nextBoard):
                        continue

                    heuristic = line_completion_heuristic(nextBoard)
                    score = (HEURISTIC_WEIGHT * heuristic) + (
                        current_model_action_weight * q_value
                    )
                    scored_actions.append(
                        (
                            score,
                            q_value,
                            heuristic,
                            int(action_index),
                            move_name,
                            nextBoard,
                        )
                    )

                scored_actions.sort(key=ranked_score, reverse=True)
                for score, q_value, heuristic, action_index, move_name, nextBoard in scored_actions:
                    next_visited_config_positions = copy_visited_config_positions(
                        visited_config_positions
                    )
                    remember_agent_position_for_config(
                        nextBoard,
                        next_visited_config_positions,
                    )

                    candidates.append(
                        (
                            score,
                            q_value,
                            heuristic,
                            nextBoard,
                            moves + move_name,
                            next_depth,
                            next_visited_config_positions,
                        )
                    )
                    best_depth_seen[nextBoard] = next_depth

            candidates.sort(key=ranked_score, reverse=True)
            kept_candidates = candidates[:beam_width]
            pruned_candidates = candidates[beam_width:]
            for candidate in pruned_candidates:
                add_to_backup(candidate)
            if debug:
                logger.debug(
                    "Depth %s: frontier=%s, candidates=%s, kept=%s, "
                    "pruned_to_backup=%s, backup=%s, seen=%s, states_checked=%s",
                    depth,
                    depth_size,
                    len(candidates),
                    len(kept_candidates),
                    len(pruned_candidates),
                    len(backup_frontier),
                    len(best_depth_seen),
                    statesChecked,
                )
            currentBoards.extend(
                (board, moves, depth, visited_config_positions)
                for _, _, _, board, moves, depth, visited_config_positions
                in kept_candidates
            )
            if not currentBoards:
                refilled = refill_from_backup()
                if debug and refilled:
                    logger.debug(
                        "Refilled active frontier from backup: %s, backup_remaining=%s",
                        refilled,
                        len(backup_frontier),
                    )
    
        logger.info(
            "beam_search.restart_failed restart=%s states_checked=%s",
            restart_index,
            statesChecked,
        )
        return "", []


    base_seed = seed if seed is not None else random.randrange(2**32)
    attempts = max(1, restarts)
    last_result = ("", [])
    for restart_index in range(attempts):
        restart_seed = base_seed + restart_index
        if restart_model_action_weights:
            weight_index = min(restart_index, len(restart_model_action_weights) - 1)
            current_model_action_weight = restart_model_action_weights[weight_index]
        else:
            current_model_action_weight = model_action_weight
        if debug:
            logger.debug(
                "Restart %s: model_action_weight=%s",
                restart_index,
                current_model_action_weight,
            )
        result = solve(
            initial_board,
            restart_seed=restart_seed,
            restart_index=restart_index,
            current_model_action_weight=current_model_action_weight,
        )
        moves, transition_data = result
        if moves:
            return result
        last_result = result

    return last_result
