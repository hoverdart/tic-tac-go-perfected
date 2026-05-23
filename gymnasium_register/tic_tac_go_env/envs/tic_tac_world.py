import gymnasium as gym
import pygame
import numpy as np
from typing import Optional

class TicTacWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, length=8, width=8, board = tuple(tuple()), render_mode=None, reset_option=8):
        self.length = length
        self.width = width
        self.board = board
        self.base_board = tuple(tuple(row) for row in board)
        self.initial_board = tuple(tuple(row) for row in board)
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None
        self.reset_option = reset_option

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        self.agentX = -1
        self.agentY = -1
        self.oOneX = -1
        self.oOneY = -1
        self.oTwoX = -1
        self.oTwoY = -1
        self.initialAgentX = -1
        self.initialAgentY = -1
        self.initialOOneX = -1
        self.initialOOneY = -1
        self.initialOTwoX = -1
        self.initialOTwoY = -1

        try:
            for i in range(0, len(board)):
                for j in range(0, len(board[i])):
                    if board[i][j] == "U":
                        self.agentX = i
                        self.agentY = j
                        self.initialAgentX = i
                        self.initialAgentY = j
                        raise StopIteration
        except StopIteration:
            pass
        
        try:
            for i in range(0, len(board)):
                for j in range(0, len(board[i])):
                    if board[i][j] == "O" and self.oTwoX == -1:
                        self.oOneX = i
                        self.oOneY = j
                        self.initialOOneX = i
                        self.initialOOneY = j
                    if board[i][j] == "O" and self.oTwoX != -1:
                        self.oTwoX = i
                        self.oTwoY = j
                        self.initialOTwoX = i
                        self.initialOTwoY = j
                        raise StopIteration
        except StopIteration:
            pass


        self.agent_location = np.array([self.agentX, self.agentY], dtype=np.int32)
        self.o_one_location = np.array([self.oOneX, self.oOneY], dtype=np.int32)
        self.o_two_location = np.array([self.oTwoX, self.oTwoY], dtype=np.int32)
        
        self.observation_space = gym.spaces.Box(low = 0, 
                                                high = 1,
                                                shape=(5, 8, 8), 
                                                dtype=np.int32)

        self.action_space = gym.spaces.Discrete(4)
    
    def moveUp(self, board=None):
        if board is None:
            board = self.board
        userPos = self.agent_location.copy()
        
        if userPos[0]-1 >= 0 :
            if board[userPos[0]-1][userPos[1]] == "B" :
                return board, self.agent_location, self.o_one_location, self.o_two_location
            elif((board[userPos[0]-1][userPos[1]] == "X" or 
                board[userPos[0]-1][userPos[1]] == "O")
                and (
                    userPos[0]-2 < 0 or board[userPos[0]-2][userPos[1]] != "")):
                return board, self.agent_location, self.o_one_location, self.o_two_location
            else:

                newBoard = [list(row) for row in board]
                if(userPos[0]-2 >= 0 and board[userPos[0]-1][userPos[1]] == "X"):
                    newBoard[userPos[0]-2][userPos[1]] = newBoard[userPos[0]-1][userPos[1]]

                elif(userPos[0]-2 >= 0 and board[userPos[0]-1][userPos[1]] == "O"):
                    if(userPos[0]-1 == self.o_one_location[0] and userPos[1] == self.o_one_location[1]):
                        self.o_one_location[0] = self.o_one_location[0] - 1
                    elif(userPos[0]-1 == self.o_two_location[0] and userPos[1] == self.o_two_location[1]):
                        self.o_two_location[0] = self.o_two_location[0] - 1

                    newBoard[userPos[0]-2][userPos[1]] = newBoard[userPos[0]-1][userPos[1]]

                newBoard[userPos[0]-1][userPos[1]] = "U"
                self.agent_location[0] = self.agent_location[0] - 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self.agent_location, self.o_one_location, self.o_two_location
        return board, self.agent_location, self.o_one_location, self.o_two_location

    def moveRight(self, board=None):
        if board is None:
            board = self.board
        userPos = self.agent_location.copy()
        
        if userPos[1]+1 < len(board[0]):
            if board[userPos[0]][userPos[1]+1] == "B":
                return board, self.agent_location, self.o_one_location, self.o_two_location
            elif ((board[userPos[0]][userPos[1]+1] == "X" or 
                board[userPos[0]][userPos[1]+1] == "O")
                and (userPos[1]+2 >= len(board[0]) or board[userPos[0]][userPos[1]+2] != "")):
                return board, self.agent_location, self.o_one_location, self.o_two_location
            else:
                newBoard = [list(row) for row in board]
                
                if (userPos[1]+2 < len(board[0]) and board[userPos[0]][userPos[1]+1] == "X"):
                    newBoard[userPos[0]][userPos[1]+2] = newBoard[userPos[0]][userPos[1]+1]

                elif (userPos[1]+2 < len(board[0]) and board[userPos[0]][userPos[1]+1] == "O"):
                    if (userPos[0] == self.o_one_location[0] and userPos[1]+1 == self.o_one_location[1]):
                        self.o_one_location[1] = self.o_one_location[1] + 1
                    elif (userPos[0] == self.o_two_location[0] and userPos[1]+1 == self.o_two_location[1]):
                        self.o_two_location[1] = self.o_two_location[1] + 1

                    newBoard[userPos[0]][userPos[1]+2] = newBoard[userPos[0]][userPos[1]+1]

                newBoard[userPos[0]][userPos[1]+1] = "U"
                self.agent_location[1] = self.agent_location[1] + 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self.agent_location, self.o_one_location, self.o_two_location
                
        return board, self.agent_location, self.o_one_location, self.o_two_location

    def moveLeft(self, board=None):
        if board is None:
            board = self.board
        userPos = self.agent_location.copy()
        
        if userPos[1]-1 >= 0:
            if board[userPos[0]][userPos[1]-1] == "B":
                return board, self.agent_location, self.o_one_location, self.o_two_location
            elif ((board[userPos[0]][userPos[1]-1] == "X" or 
                board[userPos[0]][userPos[1]-1] == "O")
                and (userPos[1]-2 < 0 or board[userPos[0]][userPos[1]-2] != "")):
                return board, self.agent_location, self.o_one_location, self.o_two_location
            else:
                newBoard = [list(row) for row in board]
                
                if (userPos[1]-2 >= 0 and board[userPos[0]][userPos[1]-1] == "X"):
                    newBoard[userPos[0]][userPos[1]-2] = newBoard[userPos[0]][userPos[1]-1]

                elif (userPos[1]-2 >= 0 and board[userPos[0]][userPos[1]-1] == "O"):
                    if (userPos[0] == self.o_one_location[0] and userPos[1]-1 == self.o_one_location[1]):
                        self.o_one_location[1] = self.o_one_location[1] - 1
                    elif (userPos[0] == self.o_two_location[0] and userPos[1]-1 == self.o_two_location[1]):
                        self.o_two_location[1] = self.o_two_location[1] - 1

                    newBoard[userPos[0]][userPos[1]-2] = newBoard[userPos[0]][userPos[1]-1]

                newBoard[userPos[0]][userPos[1]-1] = "U"
                self.agent_location[1] = self.agent_location[1] - 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self.agent_location, self.o_one_location, self.o_two_location
                
        return board, self.agent_location, self.o_one_location, self.o_two_location

    def moveDown(self, board=None):
        if board is None:
            board = self.board
        userPos = self.agent_location.copy()
        
        if userPos[0]+1 < len(board):
            if board[userPos[0]+1][userPos[1]] == "B":
                return board, self.agent_location, self.o_one_location, self.o_two_location
            elif ((board[userPos[0]+1][userPos[1]] == "X" or 
                board[userPos[0]+1][userPos[1]] == "O")
                and (userPos[0]+2 >= len(board) or board[userPos[0]+2][userPos[1]] != "")):
                return board, self.agent_location, self.o_one_location, self.o_two_location
            else:
                newBoard = [list(row) for row in board]
                
                if (userPos[0]+2 < len(board) and board[userPos[0]+1][userPos[1]] == "X"):
                    newBoard[userPos[0]+2][userPos[1]] = newBoard[userPos[0]+1][userPos[1]]

                elif (userPos[0]+2 < len(board) and board[userPos[0]+1][userPos[1]] == "O"):
                    if (userPos[0]+1 == self.o_one_location[0] and userPos[1] == self.o_one_location[1]):
                        self.o_one_location[0] = self.o_one_location[0] + 1
                    elif (userPos[0]+1 == self.o_two_location[0] and userPos[1] == self.o_two_location[1]):
                        self.o_two_location[0] = self.o_two_location[0] + 1

                    newBoard[userPos[0]+2][userPos[1]] = newBoard[userPos[0]+1][userPos[1]]

                newBoard[userPos[0]+1][userPos[1]] = "U"
                self.agent_location[0] = self.agent_location[0] + 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self.agent_location, self.o_one_location, self.o_two_location
                
        return board, self.agent_location, self.o_one_location, self.o_two_location

    def lostCheck(self, board=None):
        if board is None:
            board = self.board
        for i in range(0, len(board)):
            for j in range(0, len(board[i]) - 2):
                if board[i][j] == "X" and board[i][j + 1] == "X" and board[i][j + 2] == "X":
                    return True

        for i in range(0, len(board) - 2):
            for j in range(0, len(board[i])):
                if board[i][j] == "X" and board[i + 1][j] == "X" and board[i + 2][j] == "X":
                    return True

        return False

    def solved(self, board=None):
        if board is None:
            board = self.board
        for i in range(0, len(board)):
            for j in range(0, len(board[i]) - 2):
                if board[i][j] in ("O", "U") and board[i][j + 1] in ("O", "U") and board[i][j + 2] in ("O", "U"):
                    return True

        for i in range(0, len(board) - 2):
            for j in range(0, len(board[i])):
                if board[i][j] in ("O", "U") and board[i + 1][j] in ("O", "U") and board[i + 2][j] in ("O", "U"):
                    return True

        return False

    def softLocked(self, board=None):
        if board is None:
            board = self.board

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


    def _get_obs(self):

        mapping = {"":0, "X":1, "O":2, "U":3, "B":4}
        arr = [[mapping[cell] for cell in row] for row in self.board]

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
    
    def _get_info(self):
        boardString = ""
        for row in self.board:
            boardString += " ".join([cell if cell != "" else "." for cell in row])
        return dict(board=boardString)
    
    def reset(self, seed: Optional[int] = None, options= None):
        super().reset(seed = seed)

        if(options is None):
            options = self.reset_option

        self.board = [list(row) for row in self.base_board]

        onePositions = [{"agent":(0, 0)}, 
                        {"agent":(1, 1)}, 
                        {"agent":(2, 0)}]
        
        twoPositions = [{"agent":(2, 2)}, 
                        {"agent":(4, 3)}, 
                        {"agent":(1, 0)}]
        
        threePositions = [{"agent":(3, 3), "oOne":(1, 4), "oTwo":(4,1)}, 
                        {"agent":(2, 1), "oOne":(3, 5), "oTwo":(4,3)}, 
                        {"agent":(4, 0), "oOne":(1, 2), "oTwo":(4,4)}]
        
        fourPositions = [{"agent":(3, 3), "oOne":(1, 4), "oTwo":(4, 1),
                          "xs":[(0, 4), (1, 1), (2, 5), (4, 2), (4, 4), (5, 0)]},
                         {"agent":(2, 1), "oOne":(3, 5), "oTwo":(4, 3),
                         "xs":[(0, 2), (1, 5), (2, 4), (4, 0), (5, 2)]},
                         {"agent":(4, 0), "oOne":(1, 2), "oTwo":(4, 4),
                          "xs":[(0, 5), (1, 0), (2, 3), (3, 5), (5, 1), (5, 4)]},
                         {"agent":(3, 3), "oOne":(1, 4), "oTwo":(4, 1),
                          "xs":[(0, 4), (1, 1), (2, 5), (4, 2), (4, 4), (5, 0)]}]
        
        fivePositions = [{"agent":(5, 5), "oOne":(1, 3), "oTwo":(5, 1),
                          "xs":[(0, 4), (1, 1), (2, 5), (3, 7), (4, 2), (4, 4), (6, 3), (7, 0), (7, 4), (7, 7)]},
                         {"agent":(4, 4), "oOne":(1, 5), "oTwo":(5, 2),
                          "xs":[(0, 2), (1, 1), (2, 6), (3, 0), (4, 6), (5, 4), (6, 1), (7, 5)]},
                         {"agent":(6, 5), "oOne":(2, 2), "oTwo":(5, 6),
                          "xs":[(0, 5), (1, 0), (1, 7), (3, 3), (4, 1), (4, 5), (6, 2), (7, 6)]}]
        
        sixPositions = [{"agent":(5, 5), "oOne":(1, 3), "oTwo":(5, 1),
                         "xs":[(0, 4), (1, 1), (2, 5), (3, 6), (3, 7), (4, 2), (4, 4), (5, 7), (6, 3), (6, 4), (7, 0), (7, 4), (7, 7)]},
                        {"agent":(6, 6), "oOne":(1, 4), "oTwo":(5, 2),
                         "xs":[(0, 1), (0, 6), (1, 1), (2, 0), (2, 5), (3, 5), (4, 0), (4, 3), (5, 5), (6, 2), (7, 2), (7, 6)]},
                        {"agent":(3, 4), "oOne":(1, 6), "oTwo":(6, 1),
                         "xs":[(0, 2), (1, 2), (1, 4), (2, 6), (3, 0), (3, 1), (4, 5), (5, 3), (5, 7), (6, 5), (7, 0), (7, 3)]}]
        
        sevenPositions = [{"agent":(5, 5), "oOne":(1, 3), "oTwo":(5, 1),
                           "xs":[(0, 4), (1, 1), (2, 5), (3, 7), (4, 2), (4, 4), (5, 0), (7, 0), (7, 7)],
                           "bs":[(2, 0), (2, 1), (2, 2), (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (7, 4)]},
                          {"agent":(6, 6), "oOne":(1, 5), "oTwo":(5, 2),
                           "xs":[(0, 1), (1, 1), (2, 6), (3, 0), (3, 7), (4, 3), (5, 5), (7, 1), (7, 6)],
                           "bs":[(2, 2), (2, 3), (3, 2), (5, 0), (6, 0), (6, 1)]},
                          {"agent":(4, 6), "oOne":(1, 2), "oTwo":(6, 4),
                           "xs":[(0, 5), (1, 0), (2, 4), (3, 6), (4, 1), (5, 3), (6, 6), (7, 0), (7, 7)],
                           "bs":[(2, 0), (2, 1), (3, 1), (4, 4), (4, 5), (5, 5)]}]
        
        eightPositions = [{"agent":(4, 3), "oOne":(1, 1), "oTwo":(3, 4),
                           "xs":[(0, 4), (1, 2), (2, 0), (2, 1), (2, 3), (2, 4), (3, 3), (4, 1), (4, 5)]},
                          {"agent":(2, 0), "oOne":(2, 4), "oTwo":(4, 4),
                           "xs":[(0, 2), (1, 1), (1, 4), (2, 2), (2, 3), (3, 0), (3, 4), (3, 5), (4, 1), (4, 3), (4, 5), (5, 0)]},
                          {"agent":(0, 0), "oOne":(3, 6), "oTwo":(6, 1),
                           "xs":[(0, 6), (1, 4), (1, 5), (2, 4), (3, 5), (4, 0), (4, 2), (4, 6), (5, 1), (5, 2), (6, 3), (6, 4)],
                           "bs":[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (5, 5), (5, 6), (6, 5), (6, 6)]},
                          {"agent":(7, 4), "oOne":(3, 1), "oTwo":(3, 6),
                           "xs":[(0, 3), (1, 1), (1, 3), (1, 6), (3, 0), (3, 7), (5, 2), (5, 5), (6, 3), (6, 4), (6, 6), (7, 1), (7, 7)],
                           "bs":[(2, 4), (3, 2), (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (5, 4)]},
                          {"agent":(3, 0), "oOne":(1, 3), "oTwo":(2, 2),
                           "xs":[(0, 2), (1, 1)],
                           "bs":[(2, 1)]}]

        def clear_board(clear_blocks=False, clear_os=True, clear_xs=True, clear_agent=True):
            for i in range(0, len(self.board)):
                for j in range(0, len(self.board[i])):
                    if ((clear_agent and self.board[i][j] == "U")
                        or (clear_os and self.board[i][j] == "O")
                        or (clear_xs and self.board[i][j] == "X")
                        or (clear_blocks and self.board[i][j] == "B")):
                        self.board[i][j] = ""

        def place_position(position):
            self.board[position["agent"][0]][position["agent"][1]] = "U"

            if "oOne" in position:
                self.board[position["oOne"][0]][position["oOne"][1]] = "O"
            if "oTwo" in position:
                self.board[position["oTwo"][0]][position["oTwo"][1]] = "O"

            for xPosition in position.get("xs", []):
                self.board[xPosition[0]][xPosition[1]] = "X"

            for bPosition in position.get("bs", []):
                self.board[bPosition[0]][bPosition[1]] = "B"

        if(options==1):
            num = np.random.choice(onePositions)
            clear_board(clear_os=False, clear_xs=False)
            place_position(num)
        elif(options==2):
            num = np.random.choice(twoPositions)
            clear_board(clear_os=False, clear_xs=False)
            place_position(num)
        elif(options==3):
            num = np.random.choice(threePositions)
            clear_board()
            place_position(num)
        elif(options==4):
            num = np.random.choice(fourPositions)
            clear_board()
            place_position(num)
        elif(options==5):
            num = np.random.choice(fivePositions)
            clear_board(clear_blocks=True)
            place_position(num)
        elif(options==6):
            num = np.random.choice(sixPositions)
            clear_board(clear_blocks=True)
            place_position(num)
        elif(options==7):
            num = np.random.choice(sevenPositions)
            clear_board(clear_blocks=True)
            place_position(num)
        else:
            num = np.random.choice(eightPositions)
            clear_board(clear_blocks=True)
            place_position(num)

        length = 0
        width = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if(self.board[i][j] != "B"):
                    if(i>length-1):
                        length = i+1
                    if(j>width-1):
                        width = j+1
        
        if(length == 0):
            length = len(self.board)
        if(width == 0):
            width = len(self.board[0])

        self.length = length
        self.width = width
        self.initial_board = tuple(tuple(row) for row in self.board)

        self.agentX = -1
        self.agentY = -1
        self.oOneX = -1
        self.oOneY = -1
        self.oTwoX = -1
        self.oTwoY = -1
        self.initialAgentX = -1
        self.initialAgentY = -1
        self.initialOOneX = -1
        self.initialOOneY = -1
        self.initialOTwoX = -1
        self.initialOTwoY = -1

        try:
            for i in range(0, len(self.board)):
                for j in range(0, len(self.board[i])):
                    if self.board[i][j] == "U":
                        self.agentX = i
                        self.agentY = j
                        self.initialAgentX = i
                        self.initialAgentY = j
                        raise StopIteration
        except StopIteration:
            pass
        
        try:
            for i in range(0, len(self.board)):
                for j in range(0, len(self.board[i])):
                    if self.board[i][j] == "O" and self.oTwoX == -1:
                        self.oOneX = i
                        self.oOneY = j
                        self.initialOOneX = i
                        self.initialOOneY = j
                    if self.board[i][j] == "O" and self.oTwoX != -1:
                        self.oTwoX = i
                        self.oTwoY = j
                        self.initialOTwoX = i
                        self.initialOTwoY = j
                        raise StopIteration
        except StopIteration:
            pass

        self.agent_location = np.array([self.agentX, self.agentY], dtype=np.int32)
        self.o_one_location = np.array([self.oOneX, self.oOneY], dtype=np.int32)
        self.o_two_location = np.array([self.oTwoX, self.oTwoY], dtype=np.int32)
        self.board = tuple(tuple(row) for row in self.initial_board)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):

        currentPos = self.agent_location.copy()

        if action == 0:
            self.moveUp()
        elif action == 1:
            self.moveDown()
        elif action == 2:
            self.moveLeft()
        elif action == 3:
            self.moveRight()

        same = False
        if currentPos[0] == self.agent_location[0] and currentPos[1] == self.agent_location[1]:
            same = True

        won = self.solved()
        lost = self.lostCheck()
        softLocked = self.softLocked()

        terminated = won or lost
        truncated = False

        #Negative per step scales with size
        reward = -0.5 * (16 / (self.length * self.width))

        if softLocked:
            terminated = True
            reward += -10

        if same:
            reward += -1

        if lost:
            reward += -10
        elif won:
            reward += 40
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Tic Tac Go")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((245, 245, 245))

        rows = len(self.board)
        cols = len(self.board[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            if self.render_mode == "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])
                return None

            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

        cell_size = self.window_size / max(rows, cols)
        board_width = cols * cell_size
        board_height = rows * cell_size
        x_offset = (self.window_size - board_width) / 2
        y_offset = (self.window_size - board_height) / 2

        colors = {
            "": (255, 255, 255),
            "B": (45, 45, 45),
            "X": (52, 116, 235),
            "O": (235, 88, 52),
            "U": (61, 168, 86),
        }

        for row_index, row in enumerate(self.board):
            for col_index, cell in enumerate(row):
                rect = pygame.Rect(
                    x_offset + col_index * cell_size,
                    y_offset + row_index * cell_size,
                    cell_size,
                    cell_size,
                )
                pygame.draw.rect(canvas, colors.get(cell, (230, 230, 230)), rect)
                pygame.draw.rect(canvas, (30, 30, 30), rect, width=2)

                center = rect.center
                radius = max(4, int(cell_size * 0.28))
                if cell == "X":
                    padding = cell_size * 0.25
                    pygame.draw.line(
                        canvas,
                        (255, 255, 255),
                        (rect.left + padding, rect.top + padding),
                        (rect.right - padding, rect.bottom - padding),
                        width=max(2, int(cell_size * 0.08)),
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 255, 255),
                        (rect.right - padding, rect.top + padding),
                        (rect.left + padding, rect.bottom - padding),
                        width=max(2, int(cell_size * 0.08)),
                    )
                elif cell == "O":
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 255),
                        center,
                        radius,
                        width=max(2, int(cell_size * 0.08)),
                    )
                elif cell == "U":
                    pygame.draw.circle(canvas, (255, 255, 255), center, radius)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        
        

        
