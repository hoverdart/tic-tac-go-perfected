import sys
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
BFS_PATH = ROOT / "gymnasium_register" / "BFStoTrainer.py"
spec = importlib.util.spec_from_file_location("BFStoTrainer", BFS_PATH)
BFS_module = importlib.util.module_from_spec(spec)
sys.modules["BFStoTrainer"] = BFS_module
spec.loader.exec_module(BFS_module)
BFStoTrainer = BFS_module.BFStoTrainer


position = {
    "agent": (7, 4),
    "oOne": (3, 1),
    "oTwo": (3, 6),
    "xs": [
        (0, 3),
        (1, 1),
        (1, 3),
        (1, 6),
        (3, 0),
        (3, 7),
        (5, 2),
        (5, 5),
        (6, 3),
        (6, 4),
        (6, 6),
        (7, 1),
        (7, 7),
    ],
    "bs": [
        (2, 4),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
        (4, 3),
        (4, 4),
        (5, 4),
    ],
}


def make_board(position):
    board = [["" for _ in range(8)] for _ in range(8)]

    for row, col in position.get("bs", []):
        board[row][col] = "B"

    for row, col in position.get("xs", []):
        board[row][col] = "X"

    board[position["oOne"][0]][position["oOne"][1]] = "O"
    board[position["oTwo"][0]][position["oTwo"][1]] = "O"
    board[position["agent"][0]][position["agent"][1]] = "U"

    return tuple(tuple(row) for row in board)


if __name__ == "__main__":
    board = make_board(position)
    solver = BFStoTrainer()

    print("lost at start:", solver.lostCheck(board))
    print("solved at start:", solver.solved(board))

    data = solver.solve(board)
    print("replay steps:", len(data))
