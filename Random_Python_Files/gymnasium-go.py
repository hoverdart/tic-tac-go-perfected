from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from typing import Optional

class TicTacWorldEnv(gym.Env):

    def __init__(self, length=6, height=6, board = tuple(tuple())):
        self.length = length
        self.height = height
        self.board = board

        self.agentX = -1
        self.agentY = -1
        self.oOneX = -1
        self.oOneY = -1
        self.oTwoX = -1
        self.oTwoY = -1

        try:
            for i in range(0, len(board)):
                for j in range(0, len(board[i])):
                    if board[i][j] == "U":
                        self.agentX = i
                        self.agentY = j
                        raise StopIteration
        except StopIteration:
            pass
        
        try:
            for i in range(0, len(board)):
                for j in range(0, len(board[i])):
                    if board[i][j] == "O" and self.oTwoX == -1:
                        self.oOneX = i
                        self.oOneY = j
                    if board[i][j] == "O" and self.oTwoX != -1:
                        self.oTwoX = i
                        self.oTwoY = j
                        raise StopIteration
        except StopIteration:
            pass


        self._agent_location = np.array([self.agentX, self.agentY], dtype=np.int32)
        self._o_one_location = np.array([self.oOneX, self.oOneY], dtype=np.int32)
        self._o_two_location = np.array([self.oTwoX, self.oTwoY], dtype=np.int32)
        
        self.observationSpace = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(low = np.array([0, 0]), 
                                        high = np.array([self.length-1, self.height-1]), 
                                        dtype=int), # [x, y] coordinates

                "oOne": gym.spaces.Box(low = np.array([0, 0]), 
                                        high = np.array([self.length-1, self.height-1]), 
                                        dtype=int), # [x, y] coordinates
                
                "oTwo": gym.spaces.Box(low = np.array([0, 0]), 
                                        high = np.array([self.length-1, self.height-1]), 
                                        dtype=int), # [x, y] coordinates
            }
        )

        self.actionSpace = gym.spaces.Discrete(4)
    
    def moveUp(self, board):
        userPos = self._agent_location
        
        if userPos[0]-1 >= 0 :
            if board[userPos[0]-1][userPos[1]] == "B" :
                return board, self._agent_location, self._o_one_location, self._o_two_location
            elif((board[userPos[0]-1][userPos[1]] == "X" or 
                board[userPos[0]-1][userPos[1]] == "O")
                and (
                    userPos[0]-2 < 0 or board[userPos[0]-2][userPos[1]] != "")):
                return board, self._agent_location, self._o_one_location, self._o_two_location
            else:

                newBoard = [list(row) for row in board]
                if(userPos[0]-2 >= 0 and board[userPos[0]-1][userPos[1]] == "X"):
                    newBoard[userPos[0]-2][userPos[1]] = newBoard[userPos[0]-1][userPos[1]]

                elif(userPos[0]-2 >= 0 and board[userPos[0]-1][userPos[1]] == "O"):
                    if(userPos[0]-1 == self._o_one_location[0] and userPos[1] == self._o_one_location[1]):
                        self._o_one_location[0] = self._o_one_location[0] - 1
                    elif(userPos[0]-1 == self._o_two_location[0] and userPos[1] == self._o_two_location[1]):
                        self._o_two_location[0] = self._o_two_location[0] - 1

                    newBoard[userPos[0]-2][userPos[1]] = newBoard[userPos[0]-1][userPos[1]]

                newBoard[userPos[0]-1][userPos[1]] = "U"
                self._agent_location[0] = self._agent_location[0] - 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self._agent_location, self._o_one_location, self._o_two_location
        return board, self._agent_location, self._o_one_location, self._o_two_location

    def moveRight(self, board):
        userPos = self._agent_location
        
        if userPos[1]+1 < len(board[0]):
            if board[userPos[0]][userPos[1]+1] == "B":
                return board, self._agent_location, self._o_one_location, self._o_two_location
            elif ((board[userPos[0]][userPos[1]+1] == "X" or 
                board[userPos[0]][userPos[1]+1] == "O")
                and (userPos[1]+2 >= len(board[0]) or board[userPos[0]][userPos[1]+2] != "")):
                return board, self._agent_location, self._o_one_location, self._o_two_location
            else:
                newBoard = [list(row) for row in board]
                
                if (userPos[1]+2 < len(board[0]) and board[userPos[0]][userPos[1]+1] == "X"):
                    newBoard[userPos[0]][userPos[1]+2] = newBoard[userPos[0]][userPos[1]+1]

                elif (userPos[1]+2 < len(board[0]) and board[userPos[0]][userPos[1]+1] == "O"):
                    if (userPos[0] == self._o_one_location[0] and userPos[1]+1 == self._o_one_location[1]):
                        self._o_one_location[1] = self._o_one_location[1] + 1
                    elif (userPos[0] == self._o_two_location[0] and userPos[1]+1 == self._o_two_location[1]):
                        self._o_two_location[1] = self._o_two_location[1] + 1

                    newBoard[userPos[0]][userPos[1]+2] = newBoard[userPos[0]][userPos[1]+1]

                newBoard[userPos[0]][userPos[1]+1] = "U"
                self._agent_location[1] = self._agent_location[1] + 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self._agent_location, self._o_one_location, self._o_two_location
                
        return board, self._agent_location, self._o_one_location, self._o_two_location

    def moveLeft(self, board):
        userPos = self._agent_location
        
        if userPos[1]-1 >= 0:
            if board[userPos[0]][userPos[1]-1] == "B":
                return board, self._agent_location, self._o_one_location, self._o_two_location
            elif ((board[userPos[0]][userPos[1]-1] == "X" or 
                board[userPos[0]][userPos[1]-1] == "O")
                and (userPos[1]-2 < 0 or board[userPos[0]][userPos[1]-2] != "")):
                return board, self._agent_location, self._o_one_location, self._o_two_location
            else:
                newBoard = [list(row) for row in board]
                
                if (userPos[1]-2 >= 0 and board[userPos[0]][userPos[1]-1] == "X"):
                    newBoard[userPos[0]][userPos[1]-2] = newBoard[userPos[0]][userPos[1]-1]

                elif (userPos[1]-2 >= 0 and board[userPos[0]][userPos[1]-1] == "O"):
                    if (userPos[0] == self._o_one_location[0] and userPos[1]-1 == self._o_one_location[1]):
                        self._o_one_location[1] = self._o_one_location[1] - 1
                    elif (userPos[0] == self._o_two_location[0] and userPos[1]-1 == self._o_two_location[1]):
                        self._o_two_location[1] = self._o_two_location[1] - 1

                    newBoard[userPos[0]][userPos[1]-2] = newBoard[userPos[0]][userPos[1]-1]

                newBoard[userPos[0]][userPos[1]-1] = "U"
                self._agent_location[1] = self._agent_location[1] - 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self._agent_location, self._o_one_location, self._o_two_location
                
        return board, self._agent_location, self._o_one_location, self._o_two_location

    def moveDown(self, board):
        userPos = self._agent_location
        
        if userPos[0]+1 < len(board):
            if board[userPos[0]+1][userPos[1]] == "B":
                return board, self._agent_location, self._o_one_location, self._o_two_location
            elif ((board[userPos[0]+1][userPos[1]] == "X" or 
                board[userPos[0]+1][userPos[1]] == "O")
                and (userPos[0]+2 >= len(board) or board[userPos[0]+2][userPos[1]] != "")):
                return board, self._agent_location, self._o_one_location, self._o_two_location
            else:
                newBoard = [list(row) for row in board]
                
                if (userPos[0]+2 < len(board) and board[userPos[0]+1][userPos[1]] == "X"):
                    newBoard[userPos[0]+2][userPos[1]] = newBoard[userPos[0]+1][userPos[1]]

                elif (userPos[0]+2 < len(board) and board[userPos[0]+1][userPos[1]] == "O"):
                    if (userPos[0]+1 == self._o_one_location[0] and userPos[1] == self._o_one_location[1]):
                        self._o_one_location[0] = self._o_one_location[0] + 1
                    elif (userPos[0]+1 == self._o_two_location[0] and userPos[1] == self._o_two_location[1]):
                        self._o_two_location[0] = self._o_two_location[0] + 1

                    newBoard[userPos[0]+2][userPos[1]] = newBoard[userPos[0]+1][userPos[1]]

                newBoard[userPos[0]+1][userPos[1]] = "U"
                self._agent_location[0] = self._agent_location[0] + 1
                newBoard[userPos[0]][userPos[1]] = ""
                self.board = newBoard

                return tuple(tuple(row) for row in newBoard), self._agent_location, self._o_one_location, self._o_two_location
                
        return board, self._agent_location, self._o_one_location, self._o_two_location

    def lostCheck(board):
        for i in range(0, len(board)):
            for j in range(0, len(board[i]) - 2):
                if board[i][j] == "X" and board[i][j + 1] == "X" and board[i][j + 2] == "X":
                    return True

        for i in range(0, len(board) - 2):
            for j in range(0, len(board[i])):
                if board[i][j] == "X" and board[i + 1][j] == "X" and board[i + 2][j] == "X":
                    return True

        return False

    def solved(board):
        for i in range(0, len(board)):
            for j in range(0, len(board[i]) - 2):
                if board[i][j] in ("O", "U") and board[i][j + 1] in ("O", "U") and board[i][j + 2] in ("O", "U"):
                    return True

        for i in range(0, len(board) - 2):
            for j in range(0, len(board[i])):
                if board[i][j] in ("O", "U") and board[i + 1][j] in ("O", "U") and board[i + 2][j] in ("O", "U"):
                    return True

        return False

    def _get_obs(self):
        return {"agent": self._agent_location, "oOne": self._o_one_location, "oTwo": self._o_two_location, "board":self.board}
    
    def _get_info(self):
        boardString = ""
        for row in self.board:
            boardString += " ".join([cell if cell != "" else "." for cell in row])
        return boardString
    
    def reset(self, seed: Optional[int] = None, options: Optional[int] = None):
        super().reset(seed = seed)

        self._agent_location = np.array([self.agentX, self.agentY], dtype=np.int32)
        self._o_one_location = np.array([self.oOneX, self.oOneY], dtype=np.int32)
        self._o_two_location = np.array([self.oTwoX, self.oTwoY], dtype=np.int32)
        self.stepCount = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):

        self.stepCount += 1

        if action == 0:
            self.moveUp()
        elif action == 1:
            self.moveDown()
        elif action == 2:
            self.moveLeft()
        elif action == 3:
            self.moveRight()

        won = self.solved()
        lost = self.lostCheck()

        terminated = won or lost
        truncated = False

        reward = 0
        if lost:
            reward = -5
        elif won:
            reward = 10
        else:
            reward = -0.1
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
        
        

        




