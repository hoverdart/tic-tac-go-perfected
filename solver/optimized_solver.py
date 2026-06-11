"""
Weighted A* solver for Tic Tac Go puzzles.

The puzzle is solved when three "useful" pieces (O and U) occupy any three cells
that form a horizontal or vertical line of 3. Each move is the user piece (U)
sliding any number of steps through empty cells, optionally ending with a single
push of an adjacent movable piece (O or X) one step further.

Key design decisions
--------------------
- Board state is always represented as a flat string "key" (produced by _to_key),
  never as the 2-D board tuple. This makes visited-set lookups O(1) and keeps the
  heapq comparable without custom ordering.
- `EMPTY = "."` is the canonical empty-cell character inside key strings. The 2-D
  board uses `""` for empty cells; _to_key() converts `""` → `"."` so the key is
  unambiguous and fixed-width.
- Board geometry (line definitions, neighbor relationships) is pre-computed once
  per (rows, cols, barrier_set) combination via _geometry() and cached forever with
  lru_cache. This is cheap because the board shape never changes during a solve.
- The A* weight is configurable per SOLVER_MODE: "fast" (weight=2.0, greedy),
  "hybrid" (starts at 1.35, drops to 1.0 after first solution), "exact" (weight=0.0).
"""

from __future__ import annotations

import heapq
import itertools
import os
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from solver.randomPythonFiles.superTicTacGoSolver import (
    apply_single_move,
    normalize_board,
)


# Direction vectors used throughout: (move_char, row_delta, col_delta)
DIRECTIONS = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)

# Pieces that can be physically pushed by the user sliding into them.
MOVABLE_PIECES = {"X", "O"}

# Pieces that count toward a winning line (O and the user U).
USEFUL_PIECES = {"O", "U"}

# The empty-cell sentinel used in key strings (see _to_key).
# The 2-D board uses "" for empty; we convert to "." so key strings are
# unambiguous and every cell is exactly one character.
EMPTY = "."


@dataclass(frozen=True)
class Geometry:
    """Pre-computed, immutable structural data for a specific board shape.

    Cached per (rows, cols, barrier_set) by _geometry(). All indices are
    flat (row * cols + col), matching the layout of key strings.

    Attributes:
        rows, cols: board dimensions.
        barriers: flat indices of barrier cells (never changes during a solve).
        lines: every possible 3-in-a-row triple (horizontal + vertical).
        valid_lines: subset of lines that contain no barrier cells. Barriers
            permanently block a line, so those triples can never be winning
            lines and are excluded from heuristic and win/loss checks.
        neighbors: for each cell index, a tuple of (move_char, adjacent_idx,
            landing_idx). adjacent_idx is one step in that direction;
            landing_idx is two steps (where a pushed piece would land), or -1
            if that cell is off the board.
    """

    rows: int
    cols: int
    barriers: frozenset[int]
    lines: tuple[tuple[int, int, int], ...]
    valid_lines: tuple[tuple[int, int, int], ...]
    neighbors: tuple[tuple[tuple[str, int, int], ...], ...]


@dataclass(frozen=True)
class Parent:
    """Stores the A* back-pointer for path reconstruction.

    Attributes:
        previous: key string of the parent state, or None for the start node.
        segment: the move string (e.g. "LLU") that produced this state from
            its parent. May be multiple characters for a slide + push.
    """

    previous: str | None
    segment: str


@lru_cache(maxsize=None)
def _geometry(rows: int, cols: int, barriers: frozenset[int]) -> Geometry:
    """Build and cache the Geometry for a given board shape and barrier layout.

    The lru_cache ensures this runs at most once per unique (rows, cols, barriers)
    combination, regardless of how many boards with that shape are solved. In
    practice this is called once per server process for a standard puzzle size.

    valid_lines filters out any triple that contains at least one barrier cell —
    those lines can never be completed, so they contribute nothing to the heuristic
    or win/loss detection and are safe to drop.
    """
    lines = []
    # Horizontal triples: every consecutive run of 3 cells in each row.
    for row in range(rows):
        base = row * cols
        for col in range(cols - 2):
            idx = base + col
            lines.append((idx, idx + 1, idx + 2))

    # Vertical triples: every consecutive run of 3 cells in each column.
    for row in range(rows - 2):
        for col in range(cols):
            idx = (row * cols) + col
            lines.append((idx, idx + cols, idx + (cols * 2)))

    # Drop any triple that passes through a barrier — it can never be won.
    valid_lines = tuple(line for line in lines if not any(idx in barriers for idx in line))

    # Pre-compute neighbors for every cell so move generation doesn't recompute them.
    neighbors = []
    for idx in range(rows * cols):
        row, col = divmod(idx, cols)
        cell_neighbors = []
        for move, row_delta, col_delta in DIRECTIONS:
            next_row = row + row_delta
            next_col = col + col_delta
            if 0 <= next_row < rows and 0 <= next_col < cols:
                next_idx = (next_row * cols) + next_col
                # landing_idx is where a pushed piece would end up (two steps away).
                landing_row = row + (row_delta * 2)
                landing_col = col + (col_delta * 2)
                landing_idx = -1
                if 0 <= landing_row < rows and 0 <= landing_col < cols:
                    landing_idx = (landing_row * cols) + landing_col
                cell_neighbors.append((move, next_idx, landing_idx))
        neighbors.append(tuple(cell_neighbors))

    return Geometry(
        rows=rows,
        cols=cols,
        barriers=barriers,
        lines=tuple(lines),
        valid_lines=valid_lines,
        neighbors=tuple(neighbors),
    )


def _to_key(board: tuple[tuple[str, ...], ...]) -> str:
    """Flatten a 2-D board tuple into a fixed-width string for O(1) hashing.

    Empty cells (`""` in the board) become `"."` so every cell is exactly one
    character and the string length equals rows * cols. All search logic inside
    this module operates on key strings, never on the original 2-D board.
    """
    return "".join(EMPTY if cell == "" else cell for row in board for cell in row)


def _geometry_for_board(board: tuple[tuple[str, ...], ...]) -> Geometry:
    """Derive the Geometry for a board, using its barrier positions as the cache key."""
    rows = len(board)
    cols = len(board[0])
    barriers = frozenset(
        (row * cols) + col
        for row, cells in enumerate(board)
        for col, cell in enumerate(cells)
        if cell == "B"
    )
    return _geometry(rows, cols, barriers)


def _is_solved(key: str, geometry: Geometry) -> bool:
    """Return True if any valid (barrier-free) line is filled with three useful pieces."""
    return any(
        key[first] in USEFUL_PIECES
        and key[second] in USEFUL_PIECES
        and key[third] in USEFUL_PIECES
        for first, second, third in geometry.lines
    )


def _is_lost(key: str, geometry: Geometry) -> bool:
    """Return True if three X pieces occupy any valid line — the opponent has won."""
    return any(
        key[first] == "X" and key[second] == "X" and key[third] == "X"
        for first, second, third in geometry.lines
    )


def _piece_can_move(key: str, geometry: Geometry, idx: int) -> bool:
    """Return True if the piece at idx has at least one valid push direction.

    A push is valid when: the cell adjacent to the piece (in some direction) is
    empty or occupied by U (so U can stand there after pushing), AND the cell
    two steps away is empty (so the pushed piece has somewhere to land).
    """
    for _, push_from, push_to in geometry.neighbors[idx]:
        if push_to == -1:
            continue
        if key[push_from] not in (EMPTY, "U"):
            continue
        if key[push_to] == EMPTY:
            return True
    return False


def _spot_can_become_user(key: str, geometry: Geometry, idx: int) -> bool:
    """Return True if U could legally occupy this cell via a push sequence."""
    cell = key[idx]
    return cell in (EMPTY, "U") or (cell == "X" and _piece_can_move(key, geometry, idx))


def _spot_can_become_empty(key: str, geometry: Geometry, idx: int) -> bool:
    """Return True if this cell could be vacated (empty, or an X that can be pushed out)."""
    cell = key[idx]
    return cell == EMPTY or (cell == "X" and _piece_can_move(key, geometry, idx))


def _useful_piece_can_move(key: str, geometry: Geometry, idx: int) -> bool:
    """Return True if the useful piece at idx can be pushed in at least one direction.

    Used by _soft_locked to check whether isolated O pieces are permanently stuck.
    """
    for _, push_from, push_to in geometry.neighbors[idx]:
        if push_to == -1:
            continue
        if _spot_can_become_user(key, geometry, push_from) and _spot_can_become_empty(
            key, geometry, push_to
        ):
            return True
    return False


def _soft_locked(key: str, geometry: Geometry) -> bool:
    """Detect a diagonal two-O deadlock: a heuristic pruning rule, not exhaustive.

    If exactly 2 O pieces are placed diagonally (different row AND different column)
    and neither of them can move, the board is almost certainly unsolvable — there
    is no way to bring all three useful pieces onto a line without moving at least
    one O. We prune this branch early rather than exhaustively searching it.

    This check is intentionally conservative: it only fires for the specific pattern
    of exactly 2 diagonal O pieces that are both immobile.
    """
    o_locations = [idx for idx, cell in enumerate(key) if cell == "O"]
    if len(o_locations) != 2:
        return False

    first, second = o_locations
    first_row, first_col = divmod(first, geometry.cols)
    second_row, second_col = divmod(second, geometry.cols)
    # If both O pieces share a row or column, they're not diagonal — don't prune.
    if first_row == second_row or first_col == second_col:
        return False

    # Diagonal and neither piece can move: prune this branch.
    return not any(_useful_piece_can_move(key, geometry, idx) for idx in o_locations)


def _is_dead(key: str, geometry: Geometry) -> bool:
    """Return True if victory is structurally impossible regardless of future moves.

    Two terminal conditions:
      (a) Fewer than 3 useful pieces remain on the board — can't form a line of 3.
      (b) No valid (barrier-free) lines exist on this board at all.
    """
    if sum(cell in USEFUL_PIECES for cell in key) < 3:
        return True
    return not geometry.valid_lines


def _pruned(key: str, geometry: Geometry) -> bool:
    """Return True if this board state is hopeless and the branch should be abandoned.

    Combines the three terminal/deadlock checks into a single gate used throughout
    the search loop to avoid redundant individual calls.
    """
    return _is_lost(key, geometry) or _is_dead(key, geometry) or _soft_locked(key, geometry)


def _line_score(key: str, line: tuple[int, int, int], useful: tuple[int, ...], cols: int) -> int:
    """Compute a heuristic cost for completing a specific candidate winning line.

    Lower score = easier/closer to winning. Components:
      - distance: sum of Manhattan distances from each useful piece to its nearest
        cell on this line (how far pieces need to travel).
      - occupied bonus: -3 per useful piece already on the line (pieces in place
        need no further travel, so they lower the cost).
      - blocker penalty: +4 per X piece on the line (X pieces need to be pushed
        out before this line can be won).
    """
    occupied = 0
    blockers = 0
    distance = 0

    for target in line:
        cell = key[target]
        if cell in USEFUL_PIECES:
            occupied += 1
        elif cell == "X":
            blockers += 1

        target_row, target_col = divmod(target, cols)
        distance += min(
            abs(piece_row - target_row) + abs(piece_col - target_col)
            for piece_row, piece_col in (divmod(piece, cols) for piece in useful)
        )

    return distance - (occupied * 3) + (blockers * 4)


@lru_cache(maxsize=250_000)
def _best_line_score(key: str, geometry: Geometry) -> int:
    """A* heuristic: minimum line_score over all valid lines for this board state.

    Cached per (key, geometry) pair so repeated visits to the same state (common in
    A*) don't recompute the heuristic. Returns 0 for already-solved boards, and a
    large sentinel (1_000_000) when no useful pieces or valid lines exist.

    The result is clamped to 0 (never negative) because a negative heuristic would
    make A* inadmissible.
    """
    if _is_solved(key, geometry):
        return 0

    useful = tuple(idx for idx, cell in enumerate(key) if cell in USEFUL_PIECES)
    if len(useful) < 3:
        return 1_000_000

    return max(
        0,
        min(
            (
                _line_score(key, line, useful, geometry.cols)
                for line in geometry.valid_lines
            ),
            default=1_000_000,
        ),
    )


def _heuristic(key: str, geometry: Geometry) -> int:
    """Return the A* heuristic for this state (delegates to _best_line_score)."""
    return _best_line_score(key, geometry)


def _reachable_paths(key: str, geometry: Geometry) -> dict[int, str]:
    """BFS from U's position to every empty cell it can reach without pushing.

    Returns a dict mapping target cell index → move string to reach it (e.g.
    cell 5 → "LLU"). The move string may be multiple characters because U can
    slide through multiple empty cells in a single "move" before optionally
    pushing a piece at the destination.

    The start cell maps to "" (no moves needed to stay in place), and is included
    so callers can always look up the user's current position.
    """
    start = key.index("U")
    queue = deque([start])
    paths = {start: ""}

    while queue:
        current = queue.popleft()
        current_path = paths[current]
        for move, next_idx, _ in geometry.neighbors[current]:
            if next_idx in paths or key[next_idx] != EMPTY:
                continue
            paths[next_idx] = current_path + move
            queue.append(next_idx)

    return paths


def _move_user(key: str, user_idx: int, target_idx: int) -> str:
    """Return a new key with U moved from user_idx to target_idx (no push)."""
    if user_idx == target_idx:
        return key

    cells = list(key)
    cells[user_idx] = EMPTY
    cells[target_idx] = "U"
    return "".join(cells)


def _push_from(key: str, user_idx: int, move: str, piece_idx: int, landing_idx: int) -> str | None:
    """Return a new key after U (now at user_idx) pushes the piece at piece_idx.

    The piece slides to landing_idx. U takes the piece's old cell. Returns None
    if the push is illegal (off-board, piece not movable, or landing cell occupied).
    """
    if landing_idx == -1:
        return None
    piece = key[piece_idx]
    if piece not in MOVABLE_PIECES or key[landing_idx] != EMPTY:
        return None

    cells = list(key)
    cells[user_idx] = EMPTY
    cells[piece_idx] = "U"
    cells[landing_idx] = piece
    return "".join(cells)


def _move_score(
    next_key: str,
    geometry: Geometry,
    segment: str,
    pushed_piece: str | None,
    current_score: int,
) -> tuple[int, int, int]:
    """Score a candidate move for ordering within _next_states.

    Returns a tuple that sorts better (smaller) for:
      - larger heuristic improvement (current_score - next_score)
      - pushing an O piece (bonus) over pushing X or walking only
      - shorter move strings (fewer steps is a tiebreaker)

    This ordering biases the search toward promising moves without changing
    the correctness of the A* priority queue.
    """
    next_score = _best_line_score(next_key, geometry)
    improvement = current_score - next_score
    useful_push_bonus = 1 if pushed_piece == "O" else 0
    return (-improvement, -useful_push_bonus, len(segment))


def _next_states(key: str, geometry: Geometry):
    """Generate all legal successor states from the current board state.

    Two kinds of moves are produced:
      1. Walk-only: U slides to a reachable empty cell and that alone solves the
         puzzle. These are checked separately to avoid missing a walk-win.
      2. Walk + push: U slides to any reachable cell, then pushes an adjacent
         movable piece (O or X) one step further.

    All candidates are collected, sorted by _move_score (most promising first),
    and yielded so the A* loop explores them in a good order.
    """
    paths = _reachable_paths(key, geometry)
    user_idx = key.index("U")
    candidates = []
    current_score = _best_line_score(key, geometry)

    # Walk-only wins: check if simply moving U to a reachable cell solves the board.
    for target_idx, walk_path in paths.items():
        if not walk_path:
            continue  # Skip the starting cell (no move taken).
        walked_key = _move_user(key, user_idx, target_idx)
        if _is_solved(walked_key, geometry):
            candidates.append(
                (
                    _move_score(
                        walked_key,
                        geometry,
                        walk_path,
                        None,
                        current_score,
                    ),
                    walked_key,
                    walk_path,
                )
            )

    # Walk + push: slide U to any reachable cell, then push a neighbor one step.
    for target_idx, walk_path in paths.items():
        walked_key = _move_user(key, user_idx, target_idx)
        for move, piece_idx, landing_idx in geometry.neighbors[target_idx]:
            pushed_piece = walked_key[piece_idx]
            pushed_key = _push_from(walked_key, target_idx, move, piece_idx, landing_idx)
            if pushed_key is None:
                continue
            segment = walk_path + move
            candidates.append(
                (
                    _move_score(
                        pushed_key,
                        geometry,
                        segment,
                        pushed_piece,
                        current_score,
                    ),
                    pushed_key,
                    segment,
                )
            )

    candidates.sort(key=lambda candidate: candidate[0])
    for _, next_key, segment in candidates:
        yield next_key, segment


def _reconstruct(parents: dict[str, Parent], key: str) -> str:
    """Walk the parent chain backwards from key to the start and return the full move string."""
    segments = []
    current = key
    while True:
        parent = parents[current]
        if parent.previous is None:
            break
        segments.append(parent.segment)
        current = parent.previous
    return "".join(reversed(segments))


def _replay(
    start_board: tuple[tuple[str, ...], ...],
    moves: str,
) -> tuple[tuple[str, ...], ...]:
    """Replay a move string from the start board and return the resulting board."""
    board = start_board
    for move in moves:
        board = apply_single_move(board, move)
    return board


def _validated_result(
    start_board: tuple[tuple[str, ...], ...],
    geometry: Geometry,
    moves: str,
) -> tuple[str, tuple[tuple[str, ...], ...]] | None:
    """Replay the candidate solution and confirm the final board is actually solved.

    The A* path reconstruction could theoretically produce a string that doesn't
    correspond to a valid solution (e.g. due to a bug in parent-pointer updates).
    This sanity check catches that before returning a bad answer to the caller.
    """
    final_board = _replay(start_board, moves)
    final_key = _to_key(final_board)
    if _is_solved(final_key, geometry):
        return moves, final_board
    return None


def _weight_for_mode(mode: str) -> float:
    """Return the initial A* weight for the given mode.

    weight=2.0 (fast): very greedy — explores far fewer states but may miss
        shorter solutions.
    weight=1.35 (hybrid): starts greedy for speed, then the main loop drops it
        to 1.0 after finding the first solution to search for a shorter one.
    weight=0.0 (exact): treats A* like Dijkstra — guarantees the optimal (shortest)
        solution but explores the most states.
    """
    if mode == "fast":
        return 2.0
    if mode == "exact":
        return 0.0
    return 1.35  # hybrid default


def solve(
    start_board,
    progress_every: int = 100_000,
    max_states: int | None = None,
    mode: str | None = None,
):
    """Run weighted A* search to find a solution for the given Tic Tac Go board.

    The priority queue entry is (f_score, cost_so_far, tie_breaker, key):
      - f_score = cost_so_far + heuristic(key) * weight
      - cost_so_far is the total number of individual direction moves taken so far
        (each character in the move string counts as 1).
      - tie_breaker is an ever-increasing counter so equal-priority entries are
        dequeued FIFO rather than compared by key string.

    best_cost_seen acts as the A* closed set: if we dequeue a state with a
    cost_so_far that no longer matches the recorded best cost, it means a cheaper
    path to that state was found later — skip this stale entry.

    In hybrid mode, the weight drops from 1.35 to 1.0 after the first solution is
    found. This allows the search to continue looking for shorter solutions while
    still being biased toward promising states.

    Args:
        start_board: 2-D tuple-of-tuples (will be normalized internally).
        progress_every: print states_checked to stdout every N states (0 = silent).
        max_states: stop and return the best solution found so far after this many
            states are explored. None means no limit.
        mode: "fast", "hybrid", or "exact". Falls back to SOLVER_MODE env var,
            then "hybrid" if neither is set.

    Returns:
        (moves, final_board, states_checked) on success, or
        (None, None, states_checked) if no solution was found within budget.
    """
    start_board = normalize_board(start_board)
    geometry = _geometry_for_board(start_board)
    start_key = _to_key(start_board)
    mode = (mode or os.getenv("SOLVER_MODE") or "hybrid").strip().lower()
    if mode not in {"hybrid", "fast", "exact"}:
        mode = "hybrid"

    # Fast-path: board is already solved or immediately prunable.
    if _is_solved(start_key, geometry):
        return "", start_board, 1
    if _pruned(start_key, geometry):
        return None, None, 1

    queue = []
    counter = itertools.count()  # Tie-breaker to avoid comparing key strings in heapq.
    parents = {start_key: Parent(previous=None, segment="")}
    best_cost_seen = {start_key: 0}  # Maps key → cheapest g-cost seen so far.
    states_checked = 0
    best_solution: tuple[str, tuple[tuple[str, ...], ...], int] | None = None
    weight = _weight_for_mode(mode)

    heapq.heappush(
        queue,
        (_heuristic(start_key, geometry) * weight, 0, next(counter), start_key),
    )

    while queue:
        _priority, cost_so_far, _, current_key = heapq.heappop(queue)

        # Stale entry: a cheaper path to this state was found after this was enqueued.
        if cost_so_far != best_cost_seen.get(current_key):
            continue

        # This state can't lead to a better solution than what we already have.
        if best_solution and cost_so_far >= len(best_solution[0]):
            continue

        states_checked += 1
        if progress_every and states_checked % progress_every == 0:
            print(states_checked, flush=True)

        if _pruned(current_key, geometry):
            if max_states is not None and states_checked >= max_states:
                break
            continue

        if _is_solved(current_key, geometry):
            moves = _reconstruct(parents, current_key)
            validated = _validated_result(start_board, geometry, moves)
            if validated is not None:
                solution_moves, final_board = validated
                best_solution = (solution_moves, final_board, states_checked)
                if mode == "fast":
                    break  # Accept the first solution immediately.
                if mode == "hybrid":
                    # Relax the weight so subsequent search favors shorter paths.
                    weight = 1.0
            if max_states is not None and states_checked >= max_states:
                break
            continue

        if max_states is not None and states_checked >= max_states:
            break

        for next_key, segment in _next_states(current_key, geometry):
            if _pruned(next_key, geometry):
                continue

            next_cost = cost_so_far + len(segment)
            # Skip if this path can't beat the current best solution.
            if best_solution and next_cost >= len(best_solution[0]):
                continue
            # Skip if we've already found a cheaper way to reach next_key.
            if next_cost >= best_cost_seen.get(next_key, 1_000_000_000):
                continue

            best_cost_seen[next_key] = next_cost
            parents[next_key] = Parent(previous=current_key, segment=segment)
            next_priority = next_cost + (_heuristic(next_key, geometry) * weight)
            heapq.heappush(
                queue,
                (next_priority, next_cost, next(counter), next_key),
            )

    if best_solution is None:
        return None, None, states_checked
    return best_solution
