"""
Tic-Tac-Go Procedural Board Generator
======================================
Usage:
    from board_generator import BoardGenerator

    gen = BoardGenerator()

    # Returns list of 8x8 tuple-of-tuples boards, BFS-verified solvable
    boards = gen.generate(grad=5, count=50)
    boards = gen.generate(grad=5, count=50, seed=42, verbose=True)

    # Save as copy-pasteable Python (list of 2D tuples)
    gen.save(boards, path="grad5_boards.py", var_name="fiveBoards")

Graduation difficulty:
    1  3x3 active area, static Os, only agent randomized
    2  6x6 active area, static Os, only agent randomized
    3  8x8, randomized Os (spread out), no Xs
    4  8x8, randomized Os + sparse non-dangerous Xs
    5  8x8, full Xs
    6  8x8, dangerous near-line-threat Xs
    7  8x8, Xs + B blocks
    8  Varying sizes 4x4–6x6 (padded to 8x8 with B), dense Xs + Bs
       (8x8 excluded here — use real tournament boards for true 8x8)

Output: every board is an 8x8 tuple-of-tuples of strings.
        Cells outside the active area are filled with "B".
"""

import random
from collections import deque


BOARD_ROWS = 8
BOARD_COLS = 8


class BoardGenerator:

    # ------------------------------------------------------------------
    # Minimum required BFS solution length per graduation
    # ------------------------------------------------------------------
    _MIN_SOLUTION = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 7, 8: 6}

    # ------------------------------------------------------------------
    # Graduation parameters
    # ------------------------------------------------------------------
    # x_danger: 0=none, 1=random, 2=partial pairs, 3=near-line threats
    # align_os_bias: probability Os are placed already row/col aligned
    _GRAD_PARAMS = {
        1: dict(active_rows=3, active_cols=3, num_xs=0,  x_danger=0,
                num_bs=0, randomize_os=False, align_os_bias=1.0, min_o_dist=1),
        2: dict(active_rows=6, active_cols=6, num_xs=0,  x_danger=0,
                num_bs=0, randomize_os=False, align_os_bias=1.0, min_o_dist=2),
        3: dict(active_rows=8, active_cols=8, num_xs=0,  x_danger=0,
                num_bs=0, randomize_os=True,  align_os_bias=0.4, min_o_dist=3),
        4: dict(active_rows=8, active_cols=8, num_xs=3,  x_danger=1,
                num_bs=0, randomize_os=True,  align_os_bias=0.5, min_o_dist=3),
        5: dict(active_rows=8, active_cols=8, num_xs=8,  x_danger=1,
                num_bs=0, randomize_os=True,  align_os_bias=0.4, min_o_dist=4),
        6: dict(active_rows=8, active_cols=8, num_xs=10, x_danger=3,
                num_bs=0, randomize_os=True,  align_os_bias=0.2, min_o_dist=4),
        7: dict(active_rows=8, active_cols=8, num_xs=8,  x_danger=2,
                num_bs=4, randomize_os=True,  align_os_bias=0.2, min_o_dist=4),
        8: dict(active_rows=None, active_cols=None, num_xs=None, x_danger=2,
                num_bs=None, randomize_os=True, align_os_bias=0.3, min_o_dist=3),
    }

    # Grad 8 capped at 6x6 — 8x8 real boards are handled separately
    _GRAD8_SIZES = [
        (4, 4), (4, 5), (5, 4),
        (5, 5), (5, 6), (6, 5),
        (6, 6),
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, grad: int, count: int,
                 bfs_depth: int = 55,
                 fixed_os=None,
                 seed: int = None,
                 verbose: bool = True) -> list:
        """
        Generate `count` BFS-verified solvable boards for graduation `grad`.

        Returns:
            List of 8x8 tuple-of-tuples boards ready to drop into your env.
        """
        assert 1 <= grad <= 8, "grad must be 1-8"
        if seed is not None:
            random.seed(seed)

        results    = []
        attempts   = 0
        rej_invalid    = 0
        rej_reachable  = 0
        rej_unsolvable = 0
        rej_tooshort   = 0

        while len(results) < count:
            attempts += 1

            # 1. Generate a candidate layout
            candidate = self._generate_candidate(grad, fixed_os=fixed_os)
            if candidate is None:
                rej_invalid += 1
                continue

            board_2d = candidate["board"]   # list-of-lists

            # 2. Fast reachability pre-check — skip expensive BFS if hopeless
            if not self._os_can_align(board_2d):
                rej_reachable += 1
                continue

            # 3. BFS verify + get solution length
            key   = self._to_key(board_2d)
            sol   = self._bfs_solve(key, max_depth=bfs_depth)
            if sol is None:
                rej_unsolvable += 1
                continue
            if len(sol) < self._MIN_SOLUTION[grad]:
                rej_tooshort += 1
                continue

            results.append((self._to_key(board_2d), len(sol)))

            if verbose and len(results) % 10 == 0:
                print(f"  [grad {grad}] {len(results)}/{count}  "
                      f"attempts={attempts}  "
                      f"invalid={rej_invalid}  "
                      f"unreachable={rej_reachable}  "
                      f"unsolvable={rej_unsolvable}  "
                      f"too_short={rej_tooshort}")

        if verbose:
            lens = [s for _, s in results]
            print(f"\n  Done. {count} boards, {attempts} attempts.")
            print(f"  Solution lengths — "
                  f"min={min(lens)}  max={max(lens)}  "
                  f"avg={sum(lens)/len(lens):.1f}")

        # Return just the tuple-of-tuples boards
        return [board for board, _ in results]

    def save(self, boards: list, path: str, var_name: str = "boards"):
        """Save boards as a .py file containing a list of 2D tuples."""
        lines = [f"# {len(boards)} BFS-verified boards\n",
                 f"{var_name} = [\n"]
        for b in boards:
            lines.append("    (\n")
            for row in b:
                lines.append(f"        {row},\n")
            lines.append("    ),\n")
        lines.append("]\n")
        with open(path, "w") as f:
            f.writelines(lines)
        print(f"Saved {len(boards)} boards to {path}")

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidate(self, grad, fixed_os=None):
        """
        Build one candidate board layout.
        Returns dict with 'board' (list-of-lists) or None if invalid.
        """
        p = self._GRAD_PARAMS[grad]

        if grad == 8:
            ar, ac = random.choice(self._GRAD8_SIZES)
            area   = ar * ac
            num_xs = max(2, area // 7)
            num_bs = max(1, area // 12)
            x_danger = p["x_danger"]
        else:
            ar, ac   = p["active_rows"], p["active_cols"]
            num_xs   = p["num_xs"]
            num_bs   = p["num_bs"]
            x_danger = p["x_danger"]

        # Full 8x8 board; cells outside active area are pre-set to "B"
        board = self._empty_board(ar, ac)
        valid = [(r, c) for r in range(ar) for c in range(ac)]

        # --- Place Os ---
        if p["randomize_os"]:
            o1, o2 = self._place_os(valid, p["align_os_bias"], p["min_o_dist"])
            if o1 is None:
                return None
        else:
            if fixed_os is not None:
                o1, o2 = fixed_os
            else:
                mid = ar // 2
                o1  = (mid, 1)
                o2  = (mid, ac - 2)
            if not (self._in_active(o1, ar, ac) and
                    self._in_active(o2, ar, ac)):
                return None

        board[o1[0]][o1[1]] = "O"
        board[o2[0]][o2[1]] = "O"

        # --- Place Xs ---
        xs = []
        if num_xs > 0:
            xs = self._place_xs(board, num_xs, x_danger, ar, ac)
            if self._is_lost(board):
                return None

        # --- Place Bs inside active area ---
        bs = []
        if num_bs > 0:
            bs = self._place_bs(board, num_bs, ar, ac)

        # --- Place Agent (far from Os on harder grads) ---
        occupied = {o1, o2} | set(xs) | set(bs)
        agent = self._place_agent(board, grad, o1, o2, ar, ac, occupied)
        if agent is None:
            return None
        board[agent[0]][agent[1]] = "U"

        # --- Quick pre-BFS checks ---
        if self._is_solved(board) or self._is_lost(board):
            return None
        if self.softLocked(board):
            return None

        return {"board": board, "active": (ar, ac)}

    # ------------------------------------------------------------------
    # O placement
    # ------------------------------------------------------------------

    def _place_os(self, valid, align_bias, min_dist):
        for _ in range(500):
            o1 = random.choice(valid)
            if random.random() < align_bias:
                # Same row or col (partially aligned = easier)
                pool = (
                    [(o1[0], c) for (r, c) in valid
                     if r == o1[0] and abs(c - o1[1]) >= min_dist]
                    + [(r, o1[1]) for (r, c) in valid
                       if c == o1[1] and abs(r - o1[0]) >= min_dist]
                )
            else:
                pool = [(r, c) for (r, c) in valid
                        if abs(r-o1[0]) + abs(c-o1[1]) >= min_dist]
                if min_dist >= 3 and pool:
                    pool.sort(key=lambda p:
                              -(abs(p[0]-o1[0]) + abs(p[1]-o1[1])))
                    pool = pool[:max(1, len(pool)//2)]
            if pool:
                return o1, random.choice(pool)
        return None, None

    # ------------------------------------------------------------------
    # Agent placement
    # ------------------------------------------------------------------

    def _place_agent(self, board, grad, o1, o2, ar, ac, occupied):
        candidates = [
            (r, c) for r in range(ar) for c in range(ac)
            if board[r][c] == "" and (r, c) not in occupied
        ]
        if not candidates:
            return None
        if grad >= 5:
            # Prefer cells farthest from both Os
            candidates.sort(key=lambda p: -(
                abs(p[0]-o1[0]) + abs(p[1]-o1[1]) +
                abs(p[0]-o2[0]) + abs(p[1]-o2[1])
            ))
            candidates = candidates[:max(1, len(candidates)//3)]
        return random.choice(candidates)

    # ------------------------------------------------------------------
    # X placement strategies
    # ------------------------------------------------------------------

    def _place_xs(self, board, count, danger, ar, ac):
        if danger <= 1:
            return self._xs_random(board, count, ar, ac)
        elif danger == 2:
            return self._xs_pairs(board, count, ar, ac)
        else:
            return self._xs_threats(board, count, ar, ac)

    def _xs_random(self, board, count, ar, ac):
        xs = []
        for _ in range(count):
            c = self._random_empty(board, ar, ac)
            if c:
                board[c[0]][c[1]] = "X"
                xs.append(c)
        return xs

    def _xs_pairs(self, board, count, ar, ac):
        xs, pairs = [], count // 2
        for _ in range(pairs):
            placed = False
            for __ in range(100):
                r  = random.randint(0, ar - 1)
                c1 = random.randint(0, ac - 3)
                c2 = c1 + random.randint(1, 2)
                if c2 < ac and board[r][c1] == "" and board[r][c2] == "":
                    board[r][c1] = board[r][c2] = "X"
                    xs += [(r, c1), (r, c2)]
                    placed = True
                    break
            if not placed:
                c = self._random_empty(board, ar, ac)
                if c:
                    board[c[0]][c[1]] = "X"; xs.append(c)
        for _ in range(count - pairs * 2):
            c = self._random_empty(board, ar, ac)
            if c:
                board[c[0]][c[1]] = "X"; xs.append(c)
        return xs

    def _xs_threats(self, board, count, ar, ac):
        """X _ X near-threat patterns (one gap away from a loss)."""
        xs, threats = [], count // 2
        for _ in range(threats):
            for __ in range(200):
                if random.random() < 0.5:          # horizontal
                    r = random.randint(0, ar - 1)
                    c = random.randint(0, ac - 3)
                    if board[r][c] == "" and board[r][c+2] == "":
                        board[r][c] = board[r][c+2] = "X"
                        xs += [(r, c), (r, c+2)]; break
                else:                              # vertical
                    c = random.randint(0, ac - 1)
                    r = random.randint(0, ar - 3)
                    if board[r][c] == "" and board[r+2][c] == "":
                        board[r][c] = board[r+2][c] = "X"
                        xs += [(r, c), (r+2, c)]; break
        for _ in range(count - threats * 2):
            c = self._random_empty(board, ar, ac)
            if c:
                board[c[0]][c[1]] = "X"; xs.append(c)
        return xs

    # ------------------------------------------------------------------
    # Block placement
    # ------------------------------------------------------------------

    def _place_bs(self, board, count, ar, ac):
        bs = []
        for _ in range(count):
            c = self._random_empty(board, ar, ac)
            if c:
                board[c[0]][c[1]] = "B"; bs.append(c)
        return bs

    # ------------------------------------------------------------------
    # Reachability pre-check
    # ------------------------------------------------------------------

    def _os_can_align(self, board):
        """
        Fast flood-fill check: can the agent physically reach a cell
        adjacent to both Os (needed to push them into alignment)?
        This rejects boards where Os are walled off before BFS even starts.

        Specifically: find all cells the agent can reach via empty/pushable
        space, then check if there exists any alignment line (3 consecutive
        cells in a row or col) that contains both Os and at least one
        agent-reachable cell.
        """
        rows = len(board)
        # Find agent and O positions
        agent = None
        o_positions = []
        for r in range(rows):
            for c in range(len(board[r])):
                if board[r][c] == "U":
                    agent = (r, c)
                elif board[r][c] == "O":
                    o_positions.append((r, c))

        if agent is None or len(o_positions) != 2:
            return False

        # BFS flood fill — agent can walk on "" cells and through X/O
        # (simplified: just check empty cell connectivity for agent movement)
        reachable = set()
        q = deque([agent])
        reachable.add(agent)
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (0 <= nr < rows and 0 <= nc < len(board[nr])
                        and (nr, nc) not in reachable
                        and board[nr][nc] not in ("B",)):
                    reachable.add((nr, nc))
                    q.append((nr, nc))

        # Check if any 3-cell line contains both Os and one reachable cell
        o_set = set(o_positions)
        for r in range(rows):
            for c in range(len(board[r]) - 2):
                line = [(r, c+i) for i in range(3)]
                line_pieces = {board[p[0]][p[1]] for p in line}
                if (sum(1 for p in line if p in o_set) == 2
                        and any(p in reachable for p in line
                                if board[p[0]][p[1]] != "O")):
                    return True
        for c in range(len(board[0])):
            for r in range(rows - 2):
                line = [(r+i, c) for i in range(3)]
                if (sum(1 for p in line if p in o_set) == 2
                        and any(p in reachable for p in line
                                if board[p[0]][p[1]] != "O")):
                    return True

        # Os not yet adjacent — check if agent can reach a cell next to either O
        # so it can start pushing them toward alignment
        for or_, oc in o_positions:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = or_+dr, oc+dc
                if (0 <= nr < rows and 0 <= nc < len(board[nr])
                        and (nr, nc) in reachable):
                    return True

        return False

    # ------------------------------------------------------------------
    # Board utilities
    # ------------------------------------------------------------------

    def _empty_board(self, active_rows, active_cols):
        board = []
        for r in range(BOARD_ROWS):
            row = []
            for c in range(BOARD_COLS):
                row.append("" if (r < active_rows and c < active_cols) else "B")
            board.append(row)
        return board

    def _in_active(self, pos, ar, ac):
        return 0 <= pos[0] < ar and 0 <= pos[1] < ac

    def _random_empty(self, board, ar, ac, exclude=None):
        exclude = exclude or set()
        cells = [(r, c) for r in range(ar) for c in range(ac)
                 if board[r][c] == "" and (r, c) not in exclude]
        return random.choice(cells) if cells else None

    def _to_key(self, board):
        return tuple(tuple(row) for row in board)

    # ------------------------------------------------------------------
    # Win / loss checks
    # ------------------------------------------------------------------

    def _check_three(self, board, piece_set):
        rows = len(board)
        for r in range(rows):
            cols = len(board[r])
            for c in range(cols - 2):
                if all(board[r][c+i] in piece_set for i in range(3)):
                    return True
        for c in range(len(board[0])):
            for r in range(rows - 2):
                if all(board[r+i][c] in piece_set for i in range(3)):
                    return True
        return False

    def _is_solved(self, board):
        return self._check_three(board, {"O", "U"})

    def _is_lost(self, board):
        return self._check_three(board, {"X"})

    # ------------------------------------------------------------------
    # Softlock — your exact implementation
    # ------------------------------------------------------------------

    def softLocked(self, board=None):
        if board is None:
            return False

        def in_bounds(row, col):
            return 0 <= row < len(board) and 0 <= col < len(board[row])

        def x_is_movable(row, col):
            for row_change, col_change in [(-1,0),(1,0),(0,-1),(0,1)]:
                pfr = row - row_change; pfc = col - col_change
                ptr = row + row_change; ptc = col + col_change
                if not in_bounds(pfr, pfc): continue
                if not in_bounds(ptr, ptc): continue
                if board[pfr][pfc] not in ("", "U"): continue
                if board[ptr][ptc] == "": return True
            return False

        def spot_can_become_user(row, col):
            if board[row][col] in ("", "U"): return True
            if board[row][col] == "X": return x_is_movable(row, col)
            return False

        def spot_can_become_empty(row, col):
            if board[row][col] == "": return True
            if board[row][col] == "X": return x_is_movable(row, col)
            return False

        def o_is_movable(row, col):
            for row_change, col_change in [(-1,0),(1,0),(0,-1),(0,1)]:
                pfr = row - row_change; pfc = col - col_change
                ptr = row + row_change; ptc = col + col_change
                if not in_bounds(pfr, pfc): continue
                if not in_bounds(ptr, ptc): continue
                if not spot_can_become_user(pfr, pfc): continue
                if spot_can_become_empty(ptr, ptc): return True
            return False

        o_locations = [(i, j) for i in range(len(board))
                       for j in range(len(board[i])) if board[i][j] == "O"]

        if len(o_locations) == 2:
            first_o, second_o = o_locations
            os_are_aligned = (first_o[0] == second_o[0]
                              or first_o[1] == second_o[1])
            if not os_are_aligned:
                if (not o_is_movable(first_o[0], first_o[1])
                        and not o_is_movable(second_o[0], second_o[1])):
                    return True

        found_two_os_in_line = False

        for i in range(len(board)):
            for j in range(len(board[i]) - 2):
                line = [board[i][j], board[i][j+1], board[i][j+2]]
                if line.count("O") == 2:
                    found_two_os_in_line = True
                    for col in range(j, j+3):
                        if board[i][col] != "O" and spot_can_become_user(i, col):
                            return False

        for i in range(len(board) - 2):
            for j in range(len(board[i])):
                line = [board[i][j], board[i+1][j], board[i+2][j]]
                if line.count("O") == 2:
                    found_two_os_in_line = True
                    for row in range(i, i+3):
                        if board[row][j] != "O" and spot_can_become_user(row, j):
                            return False

        return found_two_os_in_line

    # ------------------------------------------------------------------
    # BFS solver — tuple-mutation, no deepcopy
    # ------------------------------------------------------------------

    def _legal_moves_fast(self, key):
        rows = len(key)
        agent = None
        for r in range(rows):
            for c in range(len(key[r])):
                if key[r][c] == "U":
                    agent = (r, c); break
            if agent: break
        if agent is None:
            return []

        ar, ac = agent
        moves  = []
        for action, (dr, dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
            nr, nc = ar+dr, ac+dc
            if not (0 <= nr < rows and 0 <= nc < len(key[nr])): continue
            target = key[nr][nc]
            if target == "B": continue
            if target in ("X", "O"):
                pr, pc = nr+dr, nc+dc
                if not (0 <= pr < rows and 0 <= pc < len(key[pr])): continue
                if key[pr][pc] != "": continue
                new = list(key)
                new[ar] = list(key[ar]); new[ar][ac] = ""
                new[nr] = list(key[nr]); new[nr][nc] = "U"
                new[pr] = list(key[pr]); new[pr][pc] = target
                new[ar] = tuple(new[ar])
                new[nr] = tuple(new[nr])
                new[pr] = tuple(new[pr])
                moves.append((action, tuple(new)))
            elif target == "":
                new = list(key)
                new[ar] = list(key[ar]); new[ar][ac] = ""
                new[nr] = list(key[nr]); new[nr][nc] = "U"
                new[ar] = tuple(new[ar])
                new[nr] = tuple(new[nr])
                moves.append((action, tuple(new)))
        return moves

    def _bfs_solve(self, start_key, max_depth=55):
        queue   = deque([(start_key, [])])
        visited = {start_key}
        while queue:
            key, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for action, nxt in self._legal_moves_fast(key):
                if self._is_lost(nxt):
                    continue
                if self._is_solved(nxt):
                    return path + [action]
                if nxt not in visited and not self.softLocked(nxt):
                    visited.add(nxt)
                    queue.append((nxt, path + [action]))
        return None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad",   type=int, required=True)
    parser.add_argument("--count",  type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--var",    type=str, default="boards")
    parser.add_argument("--seed",   type=int, default=None)
    args = parser.parse_args()

    gen    = BoardGenerator()
    boards = gen.generate(grad=args.grad, count=args.count,
                          seed=args.seed, verbose=True)
    if args.output:
        gen.save(boards, path=args.output, var_name=args.var)
    else:
        for b in boards[:2]:
            for row in b:
                print(row)
            print()
