"""
Ranks real Tic-Tac-Go boards by BFS solution length.
Boards that BFS can't solve within the time limit go into an "expert" pool.

Usage:
    python rank_real_boards.py

Output:
    ranked_real_boards.py  — boards sorted by solution length, split into
                             solvable (with grad suggestions) and expert pools
"""

import time
import signal
import gc
from collections import deque


# ── Paste your real boards here or import them ────────────────────────────────
# This script expects a list of (name, board) tuples where board is an
# 8x8 tuple-of-tuples. Import from your real_boards_by_grad.py or
# real_boards.py — we load all of them regardless of prior grad assignment.

TIMEOUT_SECONDS = 10   # per board — raise if you want more thorough search
BFS_MAX_DEPTH   = 80   # max solution length before giving up


# ── Game logic (mirrors your env exactly) ─────────────────────────────────────

def is_solved(key):
    rows = len(key)
    piece_set = {"O", "U"}
    for r in range(rows):
        cols = len(key[r])
        for c in range(cols - 2):
            if all(key[r][c+i] in piece_set for i in range(3)):
                return True
    for c in range(len(key[0])):
        for r in range(rows - 2):
            if all(key[r+i][c] in piece_set for i in range(3)):
                return True
    return False


def is_lost(key):
    rows = len(key)
    for r in range(rows):
        cols = len(key[r])
        for c in range(cols - 2):
            if all(key[r][c+i] == "X" for i in range(3)):
                return True
    for c in range(len(key[0])):
        for r in range(rows - 2):
            if all(key[r+i][c] == "X" for i in range(3)):
                return True
    return False


def softlocked(key):
    board = key

    def in_bounds(r, c):
        return 0 <= r < len(board) and 0 <= c < len(board[r])

    def x_movable(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            pfr, pfc = r-dr, c-dc
            ptr, ptc = r+dr, c+dc
            if not in_bounds(pfr, pfc): continue
            if not in_bounds(ptr, ptc): continue
            if board[pfr][pfc] not in ("", "U"): continue
            if board[ptr][ptc] == "": return True
        return False

    def can_become_user(r, c):
        if board[r][c] in ("", "U"): return True
        if board[r][c] == "X": return x_movable(r, c)
        return False

    def can_become_empty(r, c):
        if board[r][c] == "": return True
        if board[r][c] == "X": return x_movable(r, c)
        return False

    def o_movable(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            pfr, pfc = r-dr, c-dc
            ptr, ptc = r+dr, c+dc
            if not in_bounds(pfr, pfc): continue
            if not in_bounds(ptr, ptc): continue
            if not can_become_user(pfr, pfc): continue
            if can_become_empty(ptr, ptc): return True
        return False

    def o_can_move_direction(r, c, dr, dc):
        pfr, pfc = r-dr, c-dc
        ptr, ptc = r+dr, c+dc
        if not in_bounds(pfr, pfc): return False
        if not in_bounds(ptr, ptc): return False
        if not can_become_user(pfr, pfc): return False
        return can_become_empty(ptr, ptc)

    o_locs = [(r, c) for r in range(len(board))
              for c in range(len(board[r])) if board[r][c] == "O"]

    if len(o_locs) == 2:
        f, s = o_locs
        if f[0] != s[0] and f[1] != s[1]:
            if not o_movable(f[0], f[1]) and not o_movable(s[0], s[1]):
                return True

            left_o, right_o = sorted(o_locs, key=lambda loc: loc[1])
            if right_o[1] - left_o[1] > 2:
                if (not o_can_move_direction(left_o[0], left_o[1], 0, 1)
                        and not o_can_move_direction(right_o[0], right_o[1], 0, -1)):
                    return True

            top_o, bottom_o = sorted(o_locs, key=lambda loc: loc[0])
            if bottom_o[0] - top_o[0] > 2:
                if (not o_can_move_direction(top_o[0], top_o[1], 1, 0)
                        and not o_can_move_direction(bottom_o[0], bottom_o[1], -1, 0)):
                    return True

    found = False
    for i in range(len(board)):
        for j in range(len(board[i]) - 2):
            line = [board[i][j], board[i][j+1], board[i][j+2]]
            if line.count("O") == 2:
                found = True
                for col in range(j, j+3):
                    if board[i][col] != "O" and can_become_user(i, col):
                        return False
    for i in range(len(board) - 2):
        for j in range(len(board[i])):
            line = [board[i][j], board[i+1][j], board[i+2][j]]
            if line.count("O") == 2:
                found = True
                for row in range(i, i+3):
                    if board[row][j] != "O" and can_become_user(row, j):
                        return False
    return found


def legal_moves(key):
    rows = len(key)
    agent = None
    for r in range(rows):
        for c in range(len(key[r])):
            if key[r][c] == "U":
                agent = (r, c); break
        if agent: break
    if not agent: return []

    ar, ac = agent
    moves = []
    for action, (dr, dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
        nr, nc = ar+dr, ac+dc
        if not (0 <= nr < rows and 0 <= nc < len(key[nr])): continue
        target = key[nr][nc]
        if target == "B": continue
        if target in ("X", "O"):
            pr, pc = nr+dr, nc+dc
            if not (0 <= pr < rows and 0 <= pc < len(key[pr])): continue
            if key[pr][pc] != "": continue
            new = [list(row) for row in key]
            new[ar][ac] = ""
            new[nr][nc] = "U"
            new[pr][pc] = target
            moves.append((action, tuple(tuple(row) for row in new)))
        elif target == "":
            new = [list(row) for row in key]
            new[ar][ac] = ""
            new[nr][nc] = "U"
            moves.append((action, tuple(tuple(row) for row in new)))
    return moves


def bfs_solve(board, max_depth=BFS_MAX_DEPTH):
    start = tuple(tuple(r) for r in board)
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        key, depth = queue.popleft()
        if depth >= max_depth: continue
        for _, nxt in legal_moves(key):
            if is_lost(nxt): continue
            if is_solved(nxt): return depth + 1
            if nxt not in visited and not softlocked(nxt):
                visited.add(nxt)
                queue.append((nxt, depth + 1))
    return None


# ── Timeout wrapper ───────────────────────────────────────────────────────────

class TimeoutError(Exception): pass

def _handler(signum, frame): raise TimeoutError()

def bfs_with_timeout(board, timeout=TIMEOUT_SECONDS):
    """Returns (solution_length, timed_out)"""
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        result = bfs_solve(board)
        signal.alarm(0)
        return result, False
    except TimeoutError:
        return None, True


# ── Grade suggestion based on solution length ─────────────────────────────────

def suggest_grad(sol_len):
    if sol_len <= 5:  return 4
    if sol_len <= 13:  return 5
    if sol_len <= 19: return 6
    if sol_len <= 27: return 7
    if sol_len <= 34: return 8
    return 9


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Import all boards — combine all grad lists from your files
    # Adjust these imports to match your actual file structure
    import importlib.util, os, sys

    boards_to_rank = []  # list of (name, board_tuple)

    # Try to load from real_boards_by_grad.py in same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grad_file  = os.path.join(script_dir, "real_boards_by_grad.py")

    if os.path.exists(grad_file):
        spec = importlib.util.spec_from_file_location("grad_boards", grad_file)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        all_lists = []
        for attr in ["fourBoards","fiveBoards","sixBoards","sevenBoards","eightBoards"]:
            lst = getattr(mod, attr, [])
            all_lists.extend(lst)

        # Extract names from comments — boards are just tuples so we
        # re-parse the file to get names alongside boards
        with open(grad_file) as f:
            content = f.read()

        import re
        # Match comment lines: # Name (id)  [score=...]
        name_pattern = re.compile(r'#\s+(.+?)\s+\(\d+\)')
        names = name_pattern.findall(content)

        for i, board in enumerate(all_lists):
            name = names[i] if i < len(names) else f"Board_{i}"
            boards_to_rank.append((name, board))
    else:
        print(f"Could not find {grad_file}")
        print("Place this script in the same directory as real_boards_by_grad.py")
        sys.exit(1)

    print(f"Loaded {len(boards_to_rank)} boards to rank")
    print(f"Timeout per board: {TIMEOUT_SECONDS}s\n")

    solvable = []   # (sol_len, name, board)
    expert   = []   # (name, board)

    for i, (name, board) in enumerate(boards_to_rank):
        print(f"[{i+1:3d}/{len(boards_to_rank)}] {name:<35s}", end=" ", flush=True)
        sol_len, timed_out = bfs_with_timeout(board)
        if timed_out:
            print(f"TIMEOUT (>{TIMEOUT_SECONDS}s) → expert pool")
            expert.append((name, board))
        elif sol_len is None:
            print(f"UNSOLVABLE → expert pool")
            expert.append((name, board))
        else:
            grad = suggest_grad(sol_len)
            print(f"solved in {sol_len:2d} moves → grad {grad}")
            solvable.append((sol_len, name, board))
        gc.collect()

    solvable.sort(key=lambda x: x[0])

    print(f"\n{'='*60}")
    print(f"Solvable: {len(solvable)}  |  Expert pool: {len(expert)}")
    if solvable:
        lens = [s for s,_,_ in solvable]
        print(f"Solution lengths: min={min(lens)}  max={max(lens)}  avg={sum(lens)/len(lens):.1f}")

    # Grad distribution
    grad_counts = {4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for sol, _, _ in solvable:
        grad_counts[suggest_grad(sol)] += 1
    print("Grad distribution:", grad_counts)

    # ── Write output file ─────────────────────────────────────────────────────
    out_path = os.path.join(script_dir, "ranked_real_boards.py")

    grad_var = {4:"fourBoards", 5:"fiveBoards", 6:"sixBoards",
                7:"sevenBoards", 8:"eightBoards", 9:"nineBoards"}

    by_grad = {4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for sol, name, board in solvable:
        by_grad[suggest_grad(sol)].append((sol, name, board))

    lines = [
        "# Real boards ranked by BFS solution length\n",
        f"# Solvable: {len(solvable)}  Expert pool: {len(expert)}\n",
        f"# Grad boundaries: 4=≤5 moves  5=6-13  6=14-19  7=20-27  8=28-34  9=35+\n\n",
    ]

    for g in [4, 5, 6, 7, 8, 9]:
        boards = by_grad[g]
        lines.append(f"# Grad {g}: {len(boards)} boards\n")
        lines.append(f"{grad_var[g]} = [\n")
        for sol, name, board in boards:
            lines.append(f"    # {name}  [{sol} moves]\n")
            lines.append(f"    (\n")
            for row in board:
                lines.append(f"        {row},\n")
            lines.append(f"    ),\n")
        lines.append("]\n\n")

    lines.append("# Expert pool: BFS could not solve within time limit\n")
    lines.append("# Use these last, after mastering all grad 4-9 boards\n")
    lines.append("expertBoards = [\n")
    for name, board in expert:
        lines.append(f"    # {name}\n")
        lines.append(f"    (\n")
        for row in board:
            lines.append(f"        {row},\n")
        lines.append(f"    ),\n")
    lines.append("]\n")

    with open(out_path, "w") as f:
        f.writelines(lines)

    print(f"\nWritten to {out_path}")
