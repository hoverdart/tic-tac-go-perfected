# Tic Tac Go Solver — V1 Implementation Plan

**Goal of V1:** a classical, push-level, weighted-A\* solver that reliably solves a daily Tic Tac Go board, returns the move set to the user, and *logs the data we need to decide whether any ML tier is ever required.*

**Explicit non-goal of V1:** no learning, no policy net, no CNN, no MCTS. Those are a possible Tier 2 that V1's instrumentation will tell us whether we need. Do not build them yet.

---

## Part 1 — What Sokoban Is (full context)

Sokoban ("warehouse keeper" in Japanese) is a grid puzzle. A single **player** moves on a 4-connected grid. Some cells are **walls** (immovable). Some cells hold **boxes**. Some cells are **goals**. The player walks around and, by walking *into* a box, **pushes** it one cell in the direction of travel. The puzzle is solved when every box sits on a goal.

The rules that define the whole problem:

1. The player moves one cell at a time (up/down/left/right).
2. Walking into a box pushes it one cell forward — **but only if the cell beyond the box is empty floor.** You cannot push into a wall or into another box.
3. You can push **only one box at a time.** Two boxes in a line cannot be pushed together.
4. You cannot **pull.** This is the crucial rule.

Because you can't pull, Sokoban is full of **irreversible mistakes**. Push a box into a corner and it is stuck there forever. This single property is why Sokoban is *PSPACE-complete* and why it is one of the most-studied benchmarks in both classical search and hard-exploration RL. It is not a navigation problem; it is a problem where large regions of the state space are quietly already lost.

### The four ideas that make Sokoban solvable in practice

Every serious Sokoban solver rests on these. Our solver will too.

**(a) State = (player, box positions). Not the player's path.**
The state that matters is *where the boxes are* plus *where the player is*. The sequence of steps that got there is irrelevant.

**(b) Search over pushes, not over steps.**
This is the single most important implementation decision. Naïve search branches on the 4 player moves at every cell, and stores every walking permutation as a distinct state — that is what blows up memory. Instead, treat one **push** as the atomic search action. Between pushes, "walk the player from wherever it is to the cell needed to perform the next push" is a trivial flood-fill, not a search node. This collapses the branching factor from "4 at every cell" to "≈ (#pushable boxes) × 4 directions, minus illegal/blocked," and merges all the walk-equivalent states automatically.

**(c) Player normalization for transposition detection.**
Given a fixed box configuration, the player can freely reach every cell in its current "room" (the floor region bounded by boxes and walls). So two states with the same boxes and the player anywhere in the same reachable region are *the same state*. We canonicalize by replacing the player's position with a fixed representative of its reachable region (e.g. the minimum `(row, col)` cell reachable). This is what makes the closed set small.

**(d) Deadlock detection is often worth more than the heuristic.**
A better heuristic reaches the goal with fewer expansions. Deadlock detection prunes entire subtrees that provably can *never* reach the goal, before you waste any expansions in them. In Sokoban, cheap deadlock checks routinely matter more than a clever heuristic. Skipping them is the most common reason a "correct" solver is unusably slow.

**(e) The heuristic is a matching.**
The standard admissible heuristic is a minimum-cost assignment of boxes → goals (Hungarian algorithm) using true push-distances that respect walls. With few boxes, the matching is trivial to compute by brute force.

---

## Part 2 — Tic Tac Go **is** Sokoban (the mapping)

Tic Tac Go is a Sokoban variant. Here is the exact correspondence, plus the three things that differ and change the design.

| Sokoban | Tic Tac Go |
|---|---|
| Player | The O you control (also counts as one of the three O's for the win) |
| Boxes (must reach goals) | The **two other O's** you must line up |
| Immovable walls | Barriers / walls (block movement, cannot be pushed) |
| Movable obstacles | The **X's** — pushable one at a time, same rules as any box |
| "All boxes on goals" | **Three O's (including you) collinear and consecutive** on some line |

### The three differences that matter

**Difference 1 — the goal is *relational*, not fixed goal squares.**
Classic Sokoban has fixed goal cells. Here the goal is a *configuration*: any line (row / column / diagonal) of three consecutive cells where the two O-boxes occupy two cells and the player occupies the third. There are many candidate lines, and the "target cells" are chosen by the solver, not given. Consequence: the goal test iterates over all candidate win-lines, and the heuristic takes a `min` over them.

**A useful simplification falls out of this:** the player's *final* placement onto the third cell is just a walk (free, no push). So the only pieces whose final positions the search must engineer are the **two O's**. Once the two O's sit on two cells of some line and the third cell is empty and player-reachable, you have a solved board — append the walk and you're done. **This is effectively a 2-box Sokoban with a flexible goal.** That is a small problem.

**Difference 2 — X's are movable *and* carry a loss condition.**
X's are boxes for state-tracking purposes (pushing one changes the state and can clear a path). But creating **three X's in a row is an instant loss.** So: (i) the state must include X positions, since a push can move them; (ii) any push that produces three collinear X's is pruned as a dead/forbidden successor.

**Difference 3 — you rarely need to move most X's.**
Most X's act as static walls; solutions typically push only the one or two that block the needed lane. The search discovers this naturally (pushing an irrelevant X just costs `g` and doesn't reduce `h`), but it means the *effective* branching factor is usually much smaller than "every X × 4 directions."

### Configuration you MUST confirm before coding (these define correctness)

- **Win-line directions:** RESOLVED — **horizontal + vertical only, no diagonals.** `WIN_DIRECTIONS = {(0,1), (1,0)}`. Drop diagonal enumeration entirely.
- **Line length:** three consecutive cells (assumed here) vs a full row. Assume 3-consecutive; make it a constant `LINE_LEN = 3`.
- **Loss geometry:** RESOLVED — three X's in a row is an **automatic, immediate loss**, same geometry as the win (3 consecutive, horizontal/vertical only). Shares the win-line enumeration. Because it's fail-at-*every-step*, the prune is unconditional (never take a push that creates a 3-X line) and the verifier checks every intermediate state, not just the terminal one.
- **Board sizes:** 5×5 to 8×8, possibly irregular (barriers create non-rectangular play areas). The grid model must support arbitrary walls.
- **Cost model / "solution" definition:** the user sees keystrokes. We will *search* on push-count (for tractability) and *reconstruct* minimal keystrokes for output. Confirm nobody needs a provably keystroke-optimal solution (they don't — it's a daily puzzle, not a speedrun leaderboard). This lets us use weighted A\* and skip the optimality obligation.

---

## Part 3 — V1 Scope, Stack, Success Criteria

**Stack:** Python 3.11+. Pure standard library for the core solver (heapq, collections, dataclasses). Optional `numpy` only if profiling later shows a hot loop worth it — at push-depth ~20 it won't. This keeps it Codex-friendly and trivially testable. Package it so the eventual FastAPI endpoint can `import solve(board) -> Solution`.

**V1 is done when:**
1. It solves a corpus of real daily boards (use backfill_solutions.py, with its ALL_PAST_DAYS array containing many real boards) and returns valid keystroke solutions.
2. Every returned solution is *verified* by an independent replayer (replay keystrokes on a fresh board → reaches a win, never transiently creates 3 X's if that's a hard loss).
3. It emits a per-board instrumentation log (push-depth, nodes expanded, peak closed-set size, wall-clock).
4. On the corpus, we can read off the answer to the only open question left: **does a hard tail exist that would justify a Tier 2 ML solver?**

---

## Part 4 — Architecture

Ten modules. Each is independently testable. Build in the phase order in Part 5, not this listing order.

### 4.1 `board.py` — static board + parser
- `Board`: dimensions `R, C`; a set/bitmap of **wall** cells (immovable barriers); helper `in_bounds`, `is_wall`.
- Parser from a text format (define below) and/or JSON. Separate **static** walls from **dynamic** pieces (player, O's, X's) at parse time.
- Precompute once per board:
  - `win_lines`: list of triples of cells, each triple = 3 consecutive collinear cells in an enabled direction, fully in-bounds and containing no wall.
  - `dead_cells_for_O`: set of cells from which an O can never reach any win-line cell (Part 4.5 precompute).

**Text format (proposed):**
```
# = wall/barrier
. = empty floor
P = player (the controllable O)
O = an O box
X = an X box
```
Example 5×5:
```
#####
#P.O#
#.#X#
#..O#
#####
```
Parser returns `(Board, InitialState)`.

### 4.2 `state.py` — dynamic state + normalization
```python
@dataclass(frozen=True)
class State:
    player: int                 # canonical (normalized) player cell index
    os: frozenset[int]          # O-box cells
    xs: frozenset[int]          # X-box cells
```
- Cells encoded as `r*C + c` ints for cheap hashing.
- `normalize(player, os, xs, board) -> State`: flood-fill the player's reachable floor region (blocked by walls, O's, X's); set `player = min(reachable_region)`. This is the transposition key.
- Reachable-region flood fill is reused by push-gen and goal-test; implement once here as `reachable(from_cell, os, xs, board) -> set[int]`.

### 4.3 `moves.py` — push successor generation (PUSH LEVEL)
```
successors(state, board) -> list[(push, new_state)]
  region = reachable(state.player, state.os, state.xs, board)
  pushes = []
  for each movable piece p in state.os ∪ state.xs:
      for each direction d in {U,D,L,R}:
          stand = p - d          # cell player must occupy to push p toward d
          dest  = p + d          # cell the piece would move into
          if stand in region
             and in_bounds(dest) and not wall(dest)
             and dest not in os and dest not in xs:
                 # legal push
                 new_os, new_xs = apply_push(p, d, state)
                 if p is an X and creates_three_x_line(new_xs, board):
                     continue                      # LOSS prune (Difference 2)
                 if is_deadlock(new_os, new_xs, board):
                     continue                      # deadlock prune
                 new_state = normalize(p, new_os, new_xs, board)  # player lands on p's old cell
                 pushes.append(((piece_id, d), new_state))
  return pushes
```
Notes:
- Player after a push is on the piece's *old* cell `p`; normalization then canonicalizes it.
- Order the returned successors by heuristic (cheap, helps weighted A\* find a solution fast). Not required for correctness.

### 4.4 `goal.py` — goal test
```
is_goal(state, board) -> bool
  region = reachable(state.player, state.os, state.xs, board)  # includes state.player itself
  for triple in board.win_lines:
      os_in = [c for c in triple if c in state.os]
      if len(os_in) == 2:
          third = the one cell in triple not in os_in
          if third == state.player or (third in region and third not in os and third not in xs):
              return True
  return False
```
The `third in region` clause is what lets the free final walk close the puzzle. Record which triple + third cell satisfied it, for reconstruction.

### 4.5 `deadlock.py` — conservative deadlock detection
V1 uses only checks that are **provably safe** (never prune a solvable state). Aggressive detection is a V1.1 upgrade.

**Precompute (once per board): `dead_cells_for_O`.**
Reverse-reachability from win-line cells. An O reaches a cell by being *pushed*; reverse a push = a "pull." Starting from every cell that appears in some `win_line`, BFS backward over pull-moves using **only walls as blockers** (ignore other pieces — this *over*-approximates where an O can travel, so it *under*-approximates the dead set → safe). Every floor cell never reached is `dead_for_O`.

**Runtime checks:**
```
is_deadlock(os, xs, board) -> bool
  # (1) any O on a statically dead cell
  if any(o in board.dead_cells_for_O for o in os): return True
  # (2) frozen-O: an O blocked on BOTH axes (can't be pushed either way)
  #     AND not currently able to complete a line -> dead
  for o in os:
      if frozen_on_both_axes(o, os, xs, board) and not on_completable_line(o, os, board):
          return True
  return False
```
- `frozen_on_both_axes`: for each axis, the O can move along it only if one side is a free push-target and the other side is player-standable; if neither axis permits a push now or ever (walls on both sides of an axis), it's frozen on that axis. Frozen on both → immovable.
- Keep (2) conservative: only flag when the O is immovable by walls (not merely blocked by another movable piece, which could later clear).

**Loss check** (`creates_three_x_line`) lives in `moves.py` but shares `win_lines` geometry: after moving an X, test whether any `win_line`-geometry triple is now all-X.

### 4.6 `heuristic.py` — matching heuristic (2 boxes)
```
h(state, board) -> number   # lower bound on remaining pushes
  best = +inf
  for triple in board.win_lines:
      # skip triples already blocked by an X we can't clear cheaply? -> no, keep admissible: ignore X's
      for (cellA, cellB, cellC) as the 2 target cells for the O's and 1 for player:
          # two ways to assign the 2 O's to 2 of the 3 cells; player takes the remaining
          for assignment in the 3 choose 1 (which cell is the player's) × 2 (O pairing):
              costO = push_dist(o1, targetO1) + push_dist(o2, targetO2)
              costPlayer = 0   # walk is free in push-cost terms
              best = min(best, costO)
  return best
```
- `push_dist`: **V1 start = Manhattan distance** (ignores walls). Manhattan ≤ true push distance on a 4-grid, so it is an admissible lower bound → weighted A\* with `w=1` stays optimal, and with `w>1` stays fast. **V1.1 upgrade:** precompute true per-target push-distance maps (BFS of pulls from each candidate target) for a much tighter `h`; do this only if profiling says the frontier is too wide.
- Ignoring X-clearing cost keeps `h` admissible (it can only under-estimate). This is fine and intended.

### 4.7 `search.py` — weighted A\*
```
solve(board, initial, w=2.0, node_cap, time_cap) -> Solution | Failure
  open = heap keyed by f = g + w*h
  g[start] = 0; came_from = {}
  push start
  while open and within caps:
      s = pop-min
      if is_goal(s): return reconstruct(...)
      for (push, s2) in successors(s, board):
          ng = g[s] + 1
          if s2 unseen or ng < g[s2]:
              g[s2] = ng; came_from[s2] = (s, push)
              f = ng + w*h(s2, board); push s2
  return Failure(reason = "node_cap" | "time_cap" | "exhausted")
```
- `w` in `[1.5, 3]` for speed; expose as a param. `w=1` gives optimal (for benchmarking).
- **Caps are mandatory** and are the Tier-2 hook: if the solver ever hits a cap on a real board, that board is a Tier-2 candidate and the log will show it.
- Closed set = the `g` dict keyed by normalized `State`.

### 4.8 `reconstruct.py` — push path → keystrokes
```
reconstruct(came_from, goal_state, board) -> keystrokes[]
  1. Backtrack came_from -> ordered list of pushes [(piece, dir), ...].
  2. Replay from the initial board; for each push:
       - flood-fill shortest WALK from current player cell to the 'stand' cell (BFS over floor)
       - emit the walk keystrokes, then the push keystroke (one step in dir)
       - apply the push to the working board
  3. Final: from the satisfying triple/third-cell recorded by goal test,
     walk the player onto the third cell; emit those keystrokes.
  4. Return the full U/D/L/R sequence.
```
Also return the push-level solution (shorter, human-readable) alongside the keystrokes.

### 4.9 `verify.py` — independent solution checker (do not skip)
Replays the emitted keystrokes on a *fresh* parse of the board using a from-scratch move simulator (not the search code), and asserts: legal throughout, ends in a win, and **never creates three collinear X's at any intermediate step** (it's an automatic loss, so this is checked on every step, not just the end). This catches reconstruction bugs that the search itself can't see. Every solve in the benchmark runs through this.

### 4.10 `cli.py` / `bench.py` — entry point + instrumentation
- `cli.py`: `python -m tictacgo solve board.txt [--w 2.0]` → prints keystrokes + push summary.
- `bench.py`: runs the solver over a directory of boards and writes a CSV row per board:
  `board_id, solved(bool), push_depth, keystrokes, nodes_expanded, peak_closed_size, wall_clock_ms, w, failure_reason`.
  This CSV **is the deliverable that decides Tier 2.**

---

## Part 5 — Phased Build Order (hand to Codex phase by phase)

Each phase ends with green tests before the next starts.

**Phase 0 — Repo + data model.** Project skeleton, `board.py` parser, text format, cell encoding, `win_lines` precompute. Tests: parse round-trip; win-line enumeration on a hand-checked 5×5 for both `WIN_DIRECTIONS` settings.

**Phase 1 — Reachability + normalization.** `reachable()` flood fill; `State` + `normalize()`. Tests: reachable region on boards with boxes splitting rooms; two walk-equivalent states normalize equal; two genuinely different states don't.

**Phase 2 — Push generation.** `successors()` *without* deadlock/loss pruning yet. Tests: legal pushes on hand-built boards (box against wall = no push into wall; box behind box = no push; push updates player to old box cell). This is the correctness core — over-test it.

**Phase 3 — Goal test.** `is_goal()` including the player-reachable-third-cell rule. Tests: near-win boards (2 O's placed, third cell reachable → goal; third cell walled/blocked → not goal); diagonal cases if enabled.

**Phase 4 — Uninformed search.** Plug Phases 2–3 into a plain BFS/Dijkstra (no heuristic). Solve small boards. Tests: known-solvable boards return a path; known-unsolvable return failure. This proves the search loop before adding `h`/pruning complexity.

**Phase 5 — Loss + deadlock pruning.** `creates_three_x_line`, `dead_cells_for_O` precompute, `is_deadlock`. Tests: a push that would make 3 X's is never taken; an O pushed to a corner dead-cell is pruned; **regression guard: every previously-solvable test board is still solved** (catches over-aggressive pruning).

**Phase 6 — Heuristic + weighted A\*.** Manhattan `h`, weighted A\* with caps. Tests: `w=1` returns same solution *cost* as Phase 4's optimal search on small boards (admissibility check); `w>1` returns valid (not necessarily optimal) solutions faster.

**Phase 7 — Reconstruction + verifier.** Push path → keystrokes; independent replayer. Tests: every solved board's keystrokes verify; walk segments are shortest; final walk lands the player correctly.

**Phase 8 — CLI + benchmark harness.** Collect a real daily-board corpus, run `bench.py`, produce the CSV. **Read the CSV. Decide on Tier 2.**

**(Optional) Phase 9 — V1.1 tightening**, *only if the CSV shows a hard tail:* true push-distance heuristic maps; successor ordering by `h`; smarter freeze/corral deadlocks.

---

## Part 6 — Testing Strategy

- **Unit** per module as above.
- **Golden boards:** a folder of small hand-authored boards with known solutions and known solution costs; assert both.
- **Unsolvable boards:** boards where the O's are pre-deadlocked; assert clean failure, not a crash or infinite loop.
- **Loss-avoidance boards:** a board where the greedy path would make 3 X's; assert the solver finds the safe path.
- **Property test:** for any solved board, `verify.py` must pass. Wire this as an assertion inside `bench.py` so a bad solution fails the run loudly.
- **Regression:** the Phase-5 guard (pruning never removes a known solution) runs in CI forever.

---

## Part 7 — The Decision Gate (why Phase 8 matters most)

After Phase 8, read two columns of the CSV across the real corpus:

- **`peak_closed_size`** — if this stays small (say < a few hundred thousand) on every board including the 80-keystroke ones, the memory wall that started this project is gone. Ship V1. Tier 2 (ML) becomes an optional research track, not a requirement.
- **`nodes_expanded` / `failure_reason`** — if some boards hit the node/time cap, those are the genuine hard tail. Inspect them: are they *deep* (high `push_depth`) or *wide* (many pushable X's inflating branching)? Wide-and-shallow is exactly where a learned push-ordering prior (your original ranker, now as an MCTS/weighted-A\* prior) earns its keep. Deep-and-narrow is still classical territory (tighten `h`, add IDA\*-style fallback).

Let that CSV, not intuition, size the ML ambition.

---

## Part 8 — Open Questions to Resolve Before / During Phase 0

1. ~~`WIN_DIRECTIONS`: diagonals in scope?~~ RESOLVED — **no diagonals; horizontal + vertical only.** `WIN_DIRECTIONS = {(0,1), (1,0)}` for both win and loss geometry.
2. ~~Is "3 X's in a row" a hard loss at every step, or only at end?~~ RESOLVED — **automatic loss, checked at every step.** X-push prune is unconditional; verifier checks every intermediate state.
3. Are boards always rectangular with a wall border, or genuinely irregular? (Parser + bounds already handle irregular; just confirm the input format.)
4. Input source: will daily boards arrive as the text format above, or scraped as some JSON/DOM structure? (Write an adapter into the text format; keep the solver format-agnostic.)
5. Do we ever need *all* solutions / the shortest keystroke solution, or is one valid efficient solution enough? Assumed: one valid, reasonably short solution.

---

## Appendix — Complexity sanity check

Worst realistic board: push-depth ~20 (an 80-keystroke solution is mostly walking). Effective push branching after legal/loss/deadlock pruning: a small constant times the number of *relevant* movable pieces — typically single digits, because most X's are irrelevant scenery. That is a shallow, narrow tree. Expect thousands to low-millions of nodes in the pathological case, megabytes of closed set, sub-second to low-seconds in Python. If the benchmark contradicts this, the CSV will say so precisely — and that contradiction is itself the signal that Tier 2 is warranted.
