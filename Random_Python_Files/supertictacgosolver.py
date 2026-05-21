import time
import webbrowser
import pyautogui
from collections import deque
import datetime
import gymnasium as gym

board = [["", "", "X", "", "", ""],
         ["", "O", "", "X", "", ""],
         ["", "B", "", "X", "", "X"],
         ["X", "", "X", "", "B", ""],
         ["", "", "", "X", "O", ""],
         ["", "X", "", "", "", "U"]
         ]

currentBoards = deque()
visited = set()
currentBoards.append([tuple(tuple(row) for row in board), ""])
visited.add(currentBoards[0][0])
    
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

def moveUp(board):
    userPos = [0, 0]
    
    try:
        for i in range (0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == "U":
                    userPos = [i, j]
                    raise StopIteration
    except StopIteration:
        pass
    
    if userPos[0]-1 >= 0 :
        if board[userPos[0]-1][userPos[1]] == "B" :
            return board
        elif((board[userPos[0]-1][userPos[1]] == "X" or 
             board[userPos[0]-1][userPos[1]] == "O")
             and (
                userPos[0]-2 < 0 or board[userPos[0]-2][userPos[1]] != "")):
            return board
        else:
            newBoard = [list(row) for row in board]
            if(userPos[0]-2 >= 0 and board[userPos[0]-1][userPos[1]] != ""):
                newBoard[userPos[0]-2][userPos[1]] = newBoard[userPos[0]-1][userPos[1]]
            newBoard[userPos[0]-1][userPos[1]] = "U"
            newBoard[userPos[0]][userPos[1]] = ""
            return tuple(tuple(row) for row in newBoard)
    return board

def moveRight(board):
    userPos = [0, 0]
    
    try:
        for i in range (0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == "U":
                    userPos = [i, j]
                    raise StopIteration
    except StopIteration:
        pass
    
    if userPos[1]+1 < len(board[0]) :
        if board[userPos[0]][userPos[1]+1] == "B" :
            return board
        elif((board[userPos[0]][userPos[1]+1] == "X" or 
             board[userPos[0]][userPos[1]+1] == "O")
             and (
                userPos[1]+2 >= len(board[0]) or board[userPos[0]][userPos[1]+2] != "")):
            return board
        else:
            newBoard = [list(row) for row in board]
            if(userPos[1]+2 < len(board[0]) and board[userPos[0]][userPos[1]+1] != ""):
                newBoard[userPos[0]][userPos[1]+2] = newBoard[userPos[0]][userPos[1]+1]
            newBoard[userPos[0]][userPos[1]+1] = "U"
            newBoard[userPos[0]][userPos[1]] = ""
            return tuple(tuple(row) for row in newBoard)
    return board

def moveLeft(board):
    userPos = [0, 0]
    
    try:
        for i in range (0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == "U":
                    userPos = [i, j]
                    raise StopIteration
    except StopIteration:
        pass
    
    if userPos[1]-1 >= 0 :
        if board[userPos[0]][userPos[1]-1] == "B" :
            return board
        elif((board[userPos[0]][userPos[1]-1] == "X" or 
             board[userPos[0]][userPos[1]-1] == "O")
             and (
                userPos[1]-2 < 0 or board[userPos[0]][userPos[1]-2] != "")):
            return board
        else:
            newBoard = [list(row) for row in board]
            if(userPos[1]-2 >= 0 and board[userPos[0]][userPos[1]-1] != ""):
                newBoard[userPos[0]][userPos[1]-2] = newBoard[userPos[0]][userPos[1]-1]
            newBoard[userPos[0]][userPos[1]-1] = "U"
            newBoard[userPos[0]][userPos[1]] = ""
            return tuple(tuple(row) for row in newBoard)
    return board

def moveDown(board):
    userPos = [0, 0]
    
    try:
        for i in range (0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == "U":
                    userPos = [i, j]
                    raise StopIteration
    except StopIteration:
        pass
    
    if userPos[0]+1 < len(board) :
        if board[userPos[0]+1][userPos[1]] == "B" :
            return board
        elif((board[userPos[0]+1][userPos[1]] == "X" or 
             board[userPos[0]+1][userPos[1]] == "O")
             and (
                userPos[0]+2 >= len(board) or board[userPos[0]+2][userPos[1]] != "")):
            return board
        else:
            newBoard = [list(row) for row in board]
            if(userPos[0]+2 < len(board) and board[userPos[0]+1][userPos[1]] != ""):
                newBoard[userPos[0]+2][userPos[1]] = newBoard[userPos[0]+1][userPos[1]]
            newBoard[userPos[0]+1][userPos[1]] = "U"
            newBoard[userPos[0]][userPos[1]] = ""
            return tuple(tuple(row) for row in newBoard)
    return board

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

def printBoard(board):
    for row in board:
        print(" ".join([cell if cell != "" else "." for cell in row]))
    print()

# Replay moves step by step
def replayMoves(startBoard, moves):
    board = startBoard
    print("=== STEP BY STEP REPLAY ===")
    print("Start:")
    printBoard(board)

    for step, move in enumerate(moves, start=1):
        if move == "U":
            board = moveUp(board)
        elif move == "D":
            board = moveDown(board)
        elif move == "L":
            board = moveLeft(board)
        elif move == "R":
            board = moveRight(board)

        print("Step", step, "Move:", move)
        printBoard(board)

movesOutput = ""
statesChecked = 0
start_time = datetime.datetime.now() 

while True:
    currentBoard = currentBoards.popleft()


    # Debug: see boards being explored
    # printBoard(currentBoard[0])
    statesChecked += 1
    if statesChecked % 100000 == 0:
        print(statesChecked)

    if(lostCheck(currentBoard[0])):
        continue

    if(solved(currentBoard[0])):
        movesOutput = currentBoard[1]
        replayMoves(tuple(tuple(row) for row in board), movesOutput)
        print("=== SOLVED ===")
        print("Moves:", movesOutput)
        print("Final Board:")
        printBoard(currentBoard[0])
        print("States checked: ", statesChecked)
        break

    up = moveUp(currentBoard[0])
    down = moveDown(currentBoard[0])
    left = moveLeft(currentBoard[0])
    right = moveRight(currentBoard[0])

    if(up not in visited):
        currentBoards.append([up, currentBoard[1] + "U"])
        visited.add(up)
    if(down not in visited):
        currentBoards.append([down, currentBoard[1] + "D"])
        visited.add(down)
    if(left not in visited):
        currentBoards.append([left, currentBoard[1] + "L"])
        visited.add(left)
    if(right not in visited):
        currentBoards.append([right, currentBoard[1] + "R"])
        visited.add(right)

    if len(currentBoards) == 0:
        movesOutput = None
        print(statesChecked)
        break

end_time = datetime.datetime.now()

if(movesOutput == None):
    print("Unsolveable")
else:    
    MOVES = movesOutput

    KEY_MAP = {
        "U": "up",
        "D": "down",
        "L": "left",
        "R": "right"
    }

    webbrowser.open_new("https://www.google.com/search?q=tic+tac+go")

    time.sleep(2)

    # Click the actual Play button
    pyautogui.click(510, 531)

    # Wait for game to load
    time.sleep(1.6)

    for move in MOVES:
        pyautogui.press(KEY_MAP[move])

print("Time Taken: ", end_time - start_time)