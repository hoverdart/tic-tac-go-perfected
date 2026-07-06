# Sokoban-Style Push Solver Approach

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

Current answer: **promising, but not proven better yet.**

On the first 20 historical boards with a 10s timeout:

- Push solver solved and verified 16/20.
- Four boards timed out: `Jailbreak`, `Do Not Pass`, `Strange Fit`, `Escape Route`.

At 20s:

- Still 16/20.
- The same four boards timed out.

That tells us the core is valid, but V1 is not done as a replacement. The
failures are not "almost solved but needed a few more seconds." They are wide
search cases where the heuristic is blind through long X-clearing phases.

The push solver is probably the right architecture, but it needs stronger
classical ordering/pruning before it can replace the current production approach.

## V1 Status

V1 currently includes:

- opt-in `SOLVER_IMPL=push` routing,
- push-level weighted A*,
- player-region normalization,
- horizontal/vertical win and X-loss geometry,
- static O dead-cell pruning,
- reverse wall-aware push-distance heuristic,
- keystroke reconstruction,
- independent verifier,
- historical benchmark CLI.

The most important implementation choice in V1 is that it verifies every
returned keystroke solution independently. This matters because push-level search
and keystroke reconstruction are separate systems. A solver that finds a good
push path but emits illegal keystrokes is not useful.

## What V1 Is Missing

### 1. Obstacle-Aware X Push Ordering

The biggest missing piece is knowing which X pushes are useful.

On the timeout boards, the solver often must push X pieces many times before the
O heuristic improves. During that phase, many states have the same O-distance
score. A* then spreads across a large plateau.

V1 should add an obstacle-aware priority bias:

```text
priority = g + weight * o_heuristic + obstacle_bias
```

Useful bias features:

- Did the push increase the player's reachable region?
- Did it create more legal O pushes?
- Did it move an X off a candidate win line?
- Did it move an X off an important O push stand cell?
- Did it reduce blockers near the currently best line plan?
- Did it just shuffle an X without improving access?

This should be used only for ordering, not pruning.

### 2. Better Dynamic Heuristics

The current push-distance heuristic respects walls but mostly ignores dynamic
blockers. It knows an O could theoretically reach a line cell, but not whether
current X clusters make that path painful.

V2 should add a dynamic line-plan score:

- candidate target line,
- O-to-target assignment,
- current X blockers on target cells,
- blocked stand cells,
- unreachable stand cells,
- player-region access,
- number of useful O pushes currently available.

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

### 4. Better Benchmark Instrumentation

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

## V2 Plan

V2 should make the push solver competitive on the four timeout boards without
giving up correctness.

Recommended V2 order:

1. Add obstacle-aware priority bias.
2. Benchmark first 20 boards at 10s and 20s.
3. Confirm no regression on the existing 16 solved boards.
4. Add detailed plateau instrumentation.
5. Add top-K line-plan scoring.
6. Add optional beam fallback over push states.

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
2. Push solver with alternate weights.
3. Classical push beam fallback.
4. Current heuristic-CNN beam fallback.
5. Exact push-optimal mode for offline/storage when time allows.

The API can return the first verified solution, while offline jobs can continue
searching for a shorter push or keystroke solution.

## Recommended Direction

The push solver should remain the main research direction because it matches the
actual structure of the game. It is explainable, verifiable, and gives us a real
path to optimality.

The current production model should not be removed yet. V1 has not beaten it on
the hard tail. But V1 has already shown that a pure classical solver can solve
many real boards quickly and produce verified solutions.

The immediate next step is not more timeout. The 20s run did not change solve
rate. The immediate next step is **better X-push ordering**.

If that succeeds, the push solver becomes a strong candidate for the default
solver. If it fails, the benchmark will tell us exactly where ML or beam search
still earns its place.

