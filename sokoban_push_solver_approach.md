# Sokoban-Style Push Solver Approach

## Current V2 Notes

The current implementation keeps the Sokoban-style push abstraction as the
correctness core and adds a deterministic portfolio around it.

Important practical details:

- normal one-push successors remain complete and are always available,
- macro successors are optional strategy children and are stored as ordinary
  push tuples, so final move reconstruction and independent verification are
  unchanged,
- the portfolio shares expensive state facts across strategies instead of
  recomputing line plans and reachability from scratch,
- pair deadlock pruning is conservative: it only rejects an O-pair when both
  O pieces cannot reach any possible line assignment even under wall-only
  over-approximation,
- the deadlock layer is now X-aware in a conservative way: only pieces proven
  frozen by walls, edges, and already-proven frozen pieces become permanent
  blockers; frozen Os are allowed when they can still serve as final line
  anchors,
- batch oracle diagnostics now separate local-ordering problems from global
  search-commitment problems.
- V2.5 adds a small JSON linear ranker trained from known push-solution paths.
  The ranker scores already-legal successors only; it cannot create moves,
  prune states, or bypass verification.
- The hard-tail fallback now includes a committed beam strategy. It enumerates
  candidate final line assignments, searches each assignment with bounded beam
  restarts, uses exact transposition checks plus novelty penalties, and still
  reconstructs ordinary push sequences for independent verification.
- The JSON ranker now has two ordering signals: linear policy/value features
  and dependency-free state-action hints exported from known push paths. These
  hints are only bonuses for legal successors; they are not move replay, unsafe
  pruning, or a bypass around `verify_solution`.
- The learned feature set now includes start-state board context for hard
  tails: wall density, low-degree floor fraction, X density/count,
  X-to-wall ratio, two-X threat lines, X component count, largest X cluster,
  and X adjacency. Runtime portfolio routing uses those same broad shape
  signals for high-X boards; it no longer checks exact initial board
  signatures.

The current hard-tail work moved beyond scalar heuristic tuning. Oracle
diagnostics showed the next known-solution push was often locally near the top,
but broad X-clearing plateaus still pushed correct branches out of global
weighted queues. The committed beam fallback and hard-tail-trained policy hints
address that failure mode by committing to candidate goal assignments and giving
known-good push patterns enough ordering weight to survive bounded search. The
production coverage path remains a verified portfolio: classical strategies
first, committed/policy-guided beam fallback next, and optional heuristic-CNN
fallback where that dependency stack is available.

In the latest local hard-tail benchmark, 9 of the 14 stable CSV timeout rows
solve and verify under the same `--timeout-seconds 30 --max-nodes 500000`
command. Three of those wins are from the committed beam fallback
(`20251005`, `20251207`, `20260301`); the other wins come from the stronger
policy ordering before the fallback is needed.

The V2.9/V3.0 planning snapshot is now tracked in `tictacgo_solver_v2_plan.md`.
The full 341-board benchmark under the 30-second comparison budget is:

- Push V2.9: 326/341 verified, 95.60%.
- Beam/CNN production path: 226/341 verified, 66.28%.
- Push V2.9 plus Beam/CNN fallback: 329/341 verified, 96.48%.

Beam/CNN uniquely recovers `20260219 Untangled`, `20260701 Intersection`, and
`20260823 Spill the Beans`, so it remains useful as a narrow fallback even
though the push solver is now the stronger primary solver.

July 2026 context-ranker status:

- The no-exact-signature runtime profile preserves the focused 19-board
  recovery set at 19/19 under the 30-second benchmark command.
- The three remaining 60-second full-bank push failures are still
  `20251212 -_-`, `20251228 Cornered`, and `20260314 Tee Off`.
- The current dependency-free linear ranker can learn useful board context and
  preserve earlier hard-tail wins, but it has not solved those final three.
  The next improvement should train on frontier/search states or switch to
  FESS-style feature buckets; more exact board routing is intentionally not the
  direction.

Future work should not be another round of single priority constants. If the
pure-Python beam stalls again, the next robust layer should be either:

- a real push-state policy/value model trained from solved paths and generated
  self-play/search traces, used only for legal successor ordering and value
  estimates; or
- a FESS-style feature-bucket search that explicitly diversifies by line-plan
  progress, reachable target access, X-clearing status, and O mobility.

## Executive Summary

Tic Tac Go is best treated as a small Sokoban variant: one controllable player
piece moves around a grid and pushes movable pieces one cell at a time. That
means the solver should not primarily think in terms of individual keystrokes.
It should think in terms of **pushes**.

The new push solver approach is:

1. Collapse all walking between pushes into flood-fill reachability.
2. Search over legal pushes of `O` and `X` pieces.
3. Normalize player position by reachable region so equivalent walking states
   are stored once.
4. Use weighted A* over push states.
5. Reconstruct concrete keystrokes only after a push-level solution is found.
6. Independently verify the emitted keystrokes.

This is not "better than A*" in the sense of replacing A*. It is better framed
as **A* on the right state space**. The current optimized solver is already A*-ish
over compressed move segments, but the push solver makes the Sokoban structure
explicit and gives us better hooks for deadlocks, X-clearing, push-distance
heuristics, and optimal push solutions.

The practical goal is:

- First: produce a valid solution as fast as possible.
- Then: make the solution optimal when feasible.
- Ultimately: expose a mode split between `fast` and `optimal`, because the
  fastest solver and the optimal solver are not always the same algorithm.

## Why Sokoban Applies

Classic Sokoban has:

- a player,
- walls,
- movable boxes,
- target/goal cells,
- irreversible pushes,
- and many dead states caused by pushing boxes into bad positions.

Tic Tac Go maps naturally onto this:

| Sokoban concept | Tic Tac Go equivalent |
|---|---|
| Player | `U`, the controllable O |
| Boxes | The two non-player `O` pieces |
| Walls | `B` barriers / board edges |
| Movable obstacles | `X` pieces |
| Goal cells | Any valid 3-cell horizontal/vertical line |
| Dead pushes | X-loss lines, trapped O positions, bad obstacle rearrangements |

The key difference is that Tic Tac Go has **relational goals**. We do not know
the target cells up front. Any three consecutive horizontal/vertical cells can
become the final line. That makes the heuristic a "best target line" problem:
for every candidate line, assign the two O pieces to two cells and reserve the
third cell for the player.

The other major difference is that `X` pieces are pushable obstacles with a loss
condition. They are not goals, but moving them is often required to clear access.
This is exactly where a generic step-level search wastes time.

## Why Search Over Pushes

A keystroke-level solver branches on every `U/D/L/R` movement. Most of those
states are not strategically different. If the player can walk around a room
without pushing anything, all those player locations are equivalent for the next
push decision.

A push-level solver does this instead:

1. Flood-fill all cells the player can currently reach.
2. From that region, enumerate every piece the player can legally push.
3. Apply one push.
4. Normalize the player to a canonical representative of the new reachable
   region.

This removes a huge amount of walking noise. The solver still returns the actual
keystroke string, but it only reconstructs walking after it has found the push
sequence.

## Comparison To Existing Approaches

### Versus Old BFS-Based Solver

The old BFS-style approach is simple and robust on small boards, but it searches
too much low-level movement. It can spend a lot of effort distinguishing states
that differ only by the player's walking path.

Push solver advantages:

- Much smaller state space.
- Better transposition keys.
- Natural place for Sokoban deadlock detection.
- Easier to reason about push optimality.
- Better instrumentation: push depth, closed-set size, useful branch factor.

Push solver drawbacks:

- More complex implementation.
- Reconstruction can be buggy if not independently verified.
- Push-optimal does not automatically mean keystroke-optimal.

### Versus Current Optimized A*

The current optimized solver is already a major improvement over naive BFS. It
uses compact board keys, weighted A*, heuristic scoring, and compressed child
segments.

The push solver is not a rejection of A*. It is a more domain-specific A*:

- Current optimized solver: search over compressed movement segments.
- Push solver: search over actual Sokoban push decisions.

Push solver advantages:

- More principled state identity: `(player reachable region, O positions, X positions)`.
- Cleaner path to admissible push heuristics.
- Cleaner path to optimal push mode.
- Cleaner X-loss and deadlock pruning.
- Easier to add Sokoban-specific techniques like reverse-push distances,
  relevance cuts, tunnels, and line-plan queues.

Push solver drawbacks:

- V1 still struggles when many X pushes are required before O progress appears.
- Current heuristic is mostly O-target focused and weak on obstacle-clearing
  plateaus.
- Requires careful tie-breaking and priority shaping to avoid exploring noisy X
  rearrangements.

### Versus Beam Search + CNN

The beam-CNN pipeline is useful because it can learn patterns of promising moves
from data. It can sometimes push through boards where classical heuristics are
flat.

But beam search has tradeoffs:

- It is not naturally optimal.
- It can miss solutions outside the beam.
- It depends on model quality and training data coverage.
- It is harder to debug than a classical solver.
- It can learn shortcuts without explaining why they work.

The push solver gives us a stronger classical baseline. If it solves the real
corpus quickly, ML becomes unnecessary for production. If it fails only on a
hard tail, ML can be demoted to a narrow role: a move-ordering prior or fallback,
not the core solver.

Best long-term framing:

- Classical push solver = correctness, verification, optimal mode.
- ML/ranker = optional ordering signal for wide obstacle-clearing plateaus.
- Beam = possible fast fallback when exact-ish search is too broad.

## Is The Push Solver Better?

Current answer: **yes as a solver core, but it should remain part of a
portfolio for production.**

The earlier progress snapshot is now stale. At that point the push solver was
solving 16/20 historical boards and timing out on four. After the current
line-plan and ordering work, the first 20-board benchmark is down to two known
timeouts:

- `20250928 Jailbreak`
- `20251005 Strange Fit`

Abdullah's latest reported broader board-corpus run puts the standalone
`push_solver` at **93.8% accuracy**. Combined with the existing
`beam_search_and_cnn` path as a fallback/partner, the total reported board
coverage is now **98%**.

Those numbers should be treated as reported benchmark results until reproduced
from a checked-in benchmark command, but they change the engineering conclusion:
the push solver is no longer just a research prototype. It is a high-performing
classical solver that should be used as a primary verified path, with beam/CNN
kept for complementary hard-tail coverage.

The remaining failures still look like wide search cases rather than invalid
state modeling. The solver often needs long X-clearing phases before O progress
appears, and many frontier states have similar line-plan scores. That is why the
next step is not another single heuristic constant. The next step is a measured
portfolio: push solver first, alternate push-search schedules next, and beam/CNN
fallback where it demonstrably covers different boards.

## V1 Status

V1 currently includes:

- opt-in `SOLVER_IMPL=push` routing,
- push-level weighted A*,
- player-region normalization,
- horizontal/vertical win and X-loss geometry,
- static O dead-cell pruning,
- reverse wall-aware push-distance maps,
- precomputed floor adjacency and legal push transitions,
- push predecessors, stand cells, canonical routes, and route-cell sets,
- top-K target-line planning over O-to-target assignments,
- line-plan scoring with X blockers, player-target reachability, route cells,
  and stand cells,
- plan-specific O and X push ordering bias,
- access-gain ordering from player-region growth and legal-O-push-count growth,
- keystroke reconstruction,
- independent verifier,
- historical benchmark CLI,
- oracle-rank diagnostics for comparing successor ordering against known
  solution paths.

The most important implementation choice in V1 is that it verifies every
returned keystroke solution independently. This matters because push-level search
and keystroke reconstruction are separate systems. A solver that finds a good
push path but emits illegal keystrokes is not useful.

## Remaining Gaps

### 1. Hard-Tail Search Control

The biggest remaining problem is no longer basic X-push awareness. The current
solver already scores X pushes against target lines, O routes, stand cells, and
player targets. The remaining problem is that some boards have long plateaus
where many X pushes are plausible and none immediately improves the best O
line-plan score.

On those boards, weighted A* can still spread across too many equivalent-looking
states before it commits deeply enough to one clearing sequence.

V2 should add portfolio search rather than keep tuning one queue:

```text
baseline weighted A*
greedy weighted A*
rank-discrepancy search
macro-enabled weighted A*
policy-guided weighted A*
```

### 2. More Complete Dynamic Heuristics

The current heuristic now includes dynamic line-plan information, but it is
still mostly a local estimate. It scores a target line and its routes; it does
not fully understand multi-step clearing commitments or temporary sacrifices
where `h` gets worse before the solution becomes available.

Future scoring should track:

- whether a sequence is consistently clearing the same plan,
- whether the same X is being productively moved along a corridor,
- whether a temporary O move opens a better final line,
- whether a push improves several top-K plans at once,
- whether the solver is cycling among equally scored X shuffles.

This score does not need to be admissible if used only in `fast` mode. For
`optimal` mode, keep a separate admissible lower bound.

### 3. More Conservative Deadlock Knowledge

V1 tried wall-frozen O pruning and it was too aggressive: a stuck O can still be
a valid final anchor on a line. That prune was removed.

Future deadlock work should focus on provably safe patterns:

- static O cells that cannot reach any line cell,
- O-pair configurations from which no valid line can ever be completed,
- X-loss forced patterns,
- tiny sealed components where the player can never reach the necessary third
  cell.

Deadlock pruning is powerful, but unsafe deadlock pruning is worse than no
deadlock pruning.

### 4. Benchmark Reproducibility

The benchmark should log more than solved/timeout:

- average branching factor,
- number of X pushes expanded,
- number of O pushes expanded,
- heuristic plateau depth,
- best `h` reached before timeout,
- candidate line selected by the best frontier state,
- push-depth of returned solution,
- keystroke length,
- verifier result.

This tells us whether a failure is:

- deep but narrow,
- shallow but wide,
- X-clearing plateau,
- bad target-line selection,
- or reconstruction/verification issue.

The reported 93.8% standalone and 98% combined accuracy should be backed by a
checked-in command and output artifact. The docs should say which board corpus,
timeout, node cap, and solver order produced those numbers.

## V2 Plan

V2 should turn the current high-performing push solver into a production-grade
portfolio member without giving up correctness.

Recommended V2 order:

1. Check in the benchmark artifact that reproduces the 93.8% and 98% numbers.
2. Add a hard-tail benchmark with every board not solved by push alone.
3. Add a portfolio wrapper that can run push first and beam/CNN fallback second.
4. Add rank-discrepancy search as the next push-only fallback.
5. Add macro successors for forced corridors and target-X clearing.
6. Add a learned push ranker only after the non-ML portfolio is measurable.

### V2 Fast Mode

Fast mode should prioritize solve rate and wall-clock time:

```text
priority = g + w * static_lower_bound + obstacle_bias + line_plan_bias
```

Properties:

- not guaranteed optimal,
- should solve daily boards quickly,
- must verify every emitted solution,
- can use aggressive ordering,
- must avoid unsafe pruning.

### V2 Optimal Mode

Optimal mode should target shortest push solution first:

```text
priority = g + admissible_h
```

Properties:

- `w = 1`,
- no non-admissible bias in the primary priority,
- obstacle/line-plan scores allowed only as tie-breakers,
- returns push-optimal solution,
- may not be keystroke-optimal.

If we need keystroke-optimality, that is a different objective. A push-optimal
solution can require more walking than another solution with one extra push.

## V3 And Beyond

### 1. Multi-Queue Search

Keep a global queue plus several plan-biased queues, one per top target line.
This helps the solver commit enough effort to a plausible line plan without
fully abandoning alternatives.

### 2. Classical Beam Fallback

If A* times out, run a bounded beam over push states:

- wider but shallower,
- heavily ordered by obstacle-aware score,
- still reconstruct and verify,
- not optimal, but useful for daily solve reliability.

### 3. IDA* / Memory-Bounded Optimal Search

If closed-set memory becomes a problem, add IDA* or another memory-bounded
variant for optimal mode. This is more useful if the problem becomes deep rather
than wide.

### 4. ML As A Tie-Breaker Only

If classical V2 still has a hard tail, bring ML back narrowly:

```text
priority = classical_priority - small_weight * learned_push_score
```

ML should help order X-clearing pushes. It should not decide legality, prune
states, or replace verification.

### 5. Full Solver Portfolio

The fastest practical system may be a portfolio:

1. Push solver fast mode.
2. Push solver with alternate weights or rank-discrepancy search.
3. Macro-enabled push solver.
4. Current heuristic-CNN beam fallback.
5. Exact push-optimal mode for offline/storage when time allows.

The API can return the first verified solution, while offline jobs can continue
searching for a shorter push or keystroke solution.

## Recommended Direction

The push solver should remain the main research direction because it matches the
actual structure of the game. It is explainable, verifiable, and gives us a real
path to optimality.

The current production model should not be removed yet. The reported combined
accuracy is now 98%, which means the best system today is the portfolio, not a
single solver. The push solver should be the first verified path, and
beam/CNN should remain available for boards where its learned prior covers a
different part of the hard tail.

The immediate next step is not more timeout. The immediate next step is
**reproducible portfolio benchmarking**: record which boards push solves, which
boards beam/CNN solves, and which boards remain unsolved by both.

If the portfolio data holds, the push solver becomes the default first attempt
and the beam/CNN solver becomes a targeted fallback. If it does not hold, the
benchmark will tell us exactly where ML or beam search still earns its place.
