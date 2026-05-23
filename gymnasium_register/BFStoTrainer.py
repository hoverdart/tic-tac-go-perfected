from collections import deque
import numpy as np


class BFStoTrainer:
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

    def make_obs(self, board):
        mapping = {"": 0, "X": 1, "O": 2, "U": 3, "B": 4}
        arr = [[mapping[cell] for cell in row] for row in board]
        compatibleBoard = np.array(arr, dtype=np.int32)
        threeDArr = np.zeros((5, 8, 8), dtype=np.int32)

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

        return threeDArr

    def replayMoves(self, startBoard, moves):
        board = tuple(tuple(row) for row in startBoard)
        data = []

        for i, move in enumerate(moves):
            state_before_move = self.make_obs(board)

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

            reward_got = -0.1 * (16 / (len(startBoard) * len(startBoard[0])))
            done = i == len(moves) - 1
            if done:
                reward_got += 40

            data.append(
                {
                    "observation": state_before_move,
                    "action": action_chose,
                    "reward": reward_got,
                    "next_observation": self.make_obs(next_board),
                    "done": done,
                }
            )

            board = next_board

        return data

    def solve(self, board):
        start_board = tuple(tuple(row) for row in board)
        currentBoards = deque()
        visited = set()
        currentBoards.append((start_board, ""))
        visited.add(start_board)
        statesChecked = 0

        while currentBoards:
            currentBoard, moves = currentBoards.popleft()
            statesChecked += 1

            if statesChecked % 100000 == 0:
                print(statesChecked)

            if self.lostCheck(currentBoard):
                continue

            if self.solved(currentBoard):
                print("=== SOLVED ===")
                print("Moves:", moves)
                print("States checked: ", statesChecked)
                return self.replayMoves(start_board, moves)

            nextBoards = [
                (self.moveUp(currentBoard), moves + "U"),
                (self.moveDown(currentBoard), moves + "D"),
                (self.moveLeft(currentBoard), moves + "L"),
                (self.moveRight(currentBoard), moves + "R"),
            ]

            for nextBoard, nextMoves in nextBoards:
                if nextBoard not in visited:
                    currentBoards.append((nextBoard, nextMoves))
                    visited.add(nextBoard)

        print("No solution found. States checked:", statesChecked)
        return []
