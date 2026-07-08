# Tic Tac Go Push Solver V2 Implementation Plan

## Implementation Status

V2 is now the default `solver.push_solver.solve` path. `solve_v1` remains
available for baseline comparisons.

Implemented:

- deterministic classical portfolio: `v1_weighted`, `greedy_low_g`,
  `greedy_bias`, `macro_greedy`, `rank_discrepancy`, and
  `policy_rank_discrepancy`,
- committed beam fallback: final-assignment line plans are enumerated, then
  bounded beam restarts search those assignments with exact transposition
  filtering and novelty penalties,
- shared portfolio search context for region, heuristic, line-plan,
  O-push-count, and successor caches,
- safe O-pair deadlock pruning using wall-only push reachability,
- conservative X-aware frozen-piece deadlock constraints:
  frozen Xs are treated as permanent blockers, frozen Os must remain compatible
  with at least one possible final line, and known solution paths are checked
  against over-pruning,
- forced same-piece/same-direction macro successors that expand back into
  ordinary verified pushes,
- dependency-free V2.5 linear push ranker:
  `solver/push_solver/linear_push_ranker_v1.json`,
- hard-tail state-action policy hints exported into the same JSON artifact;
  hints only reorder legal successors and all returned move strings still go
  through `verify_solution`,
- policy feature extraction and rank-policy inference through
  `policy_features.py` and `rank_policy.py`,
- training/export command:
  `python3 -m solver.push_solver.training_export`,
- per-attempt result metadata in `PushSolveResult`,
- batch oracle-rank diagnostics through
  `python3 -m solver.push_solver.debug_oracle_rank --failed-only`,
- benchmark diagnostics columns for strategy and attempt summaries,
- opt-in benchmark beam/CNN fallback through
  `--beam-fallback --beam-timeout-seconds N`.

Measured after the initial V2 portfolio pass:

- pure push benchmark improved from `315/341` to `324/341`,
- 9 of the 26 previously failed rows were solved by V2,
- the remaining 17 pure-push rows still time out under a 30s board budget,
- shared context and macros increased node throughput on the remaining failures,
  but did not solve additional rows in the latest pure-push run,
- V2.5 policy ranking then solved `20260830 Time Capsule` with
  `policy_rank_discrepancy`, bringing pure push to `325/341`,
- the X-aware frozen-piece pruning pass is active and reduces some remaining
  search trees, but the benchmark remains `325/341` under the same 30s workflow,
- committed beam plus hard-tail state-action hints solves `20260802 Cold Feet`
  through the public `solve(...)` path in a local 30-second run, returning a
  verified 17-push / 47-keystroke solution,
- the fixed 16-board hard-tail rerun now solves 9 of the 14 stable CSV timeout
  rows, with committed-beam wins on `20251005 Strange Fit`,
  `20251207 Thread the Needle`, and `20260301 Immovable Objects`,
- full 341-board comparison under the 30-second benchmark budget:
  push solves 326/341, Beam/CNN solves 226/341, and push plus Beam/CNN fallback
  solves 329/341.

Current command set:

```bash
python3 -m unittest tests.test_push_solver tests.test_solver_service
python3 -m solver.push_solver.debug_oracle_rank --failed-only
python3 -m solver.push_solver.training_export \
  --model-out solver/push_solver/linear_push_ranker_v1.json \
  --epochs 30 --learning-rate 0.12 \
  --hard-tail-boost 4 --value-weight 1.5 --state-action-bonus 80 \
  --board-id 20250928 --board-id 20251005 --board-id 20251116 \
  --board-id 20251221 --board-id 20260208 --board-id 20260220 \
  --board-id 20260301 --board-id 20260524 --board-id 20260614 \
  --board-id 20260627 --board-id 20260802 --board-id 20260830
python3 -m solver.push_solver.benchmark_hard_tail \
  --run-solver --timeout-seconds 30 --max-nodes 500000 --weight 2.0
python3 solver/gymnasium_register/benchmark_push_solver_remaining.py \
  --only-failed --timeout-seconds 30 --max-nodes 500000 --workers 6
python3 solver/gymnasium_register/benchmark_push_solver_remaining.py \
  --only-failed --timeout-seconds 30 --max-nodes 500000 --workers 6 \
  --beam-fallback --beam-timeout-seconds 30
python3 -m solver.gymnasium_register.benchmark_push_vs_beam \
  --output debug-artifacts/push_vs_beam_full.csv \
  --summary-output debug-artifacts/push_vs_beam_full.summary.json \
  --workers 6 --push-timeout-seconds 30 --beam-timeout-seconds 30 \
  --push-max-nodes 500000 --beam-width 5000 --beam-max-depth 200
```

## V2.9/V3.0 Snapshot

V2.9 is the current pure-Python push solver. It is still a verified push-level
Sokoban solver, but the hard-tail path is now policy-guided:

- Classical portfolio first: weighted search, greedy variants, line-committed
  rank-discrepancy, macros, and policy rank-discrepancy.
- Committed beam fallback last: enumerate candidate final line assignments,
  search each assignment with bounded restarts, exact transposition filtering,
  and novelty penalties.
- Policy artifact: a dependency-free JSON ranker with linear policy/value
  weights plus state-action hints exported from known push paths.
- Safety boundary: the policy only reorders already-legal children; every move
  string still goes through `verify_solution`.

The V3.0 direction should be a real hard-tail search layer rather than more
single-number priority tuning:

- Train a push-state policy/value model from verified paths and search traces.
- Use it only for legal successor ordering, value estimates, and restart
  scheduling.
- Add a FESS-style feature-bucket search that diversifies by line-plan progress,
  player-target access, X-clearing status, O mobility, and frozen-piece risk.
- Keep the current verifier as the only acceptance gate.

## 341-Board Push vs Beam/CNN Comparison

Command used:

```bash
/tmp/tictacgo-bench-venv/bin/python -m solver.gymnasium_register.benchmark_push_vs_beam \
  --output /tmp/tictacgo_push_vs_beam_full.csv \
  --workers 6 --push-timeout-seconds 30 --beam-timeout-seconds 30 \
  --push-max-nodes 500000 --beam-width 5000 --beam-max-depth 200
cp /tmp/tictacgo_push_vs_beam_full.csv debug-artifacts/push_vs_beam_full.csv
```

For quick experiments, the comparison script also supports:

```bash
python3 -m solver.gymnasium_register.benchmark_push_vs_beam \
  --solver push --num-tests 25 --timeout-seconds 60

/tmp/tictacgo-bench-venv/bin/python -m solver.gymnasium_register.benchmark_push_vs_beam \
  --num-tests 25 --timeout-seconds 60 \
  --output /tmp/push_vs_beam_25.csv \
  --summary-output /tmp/push_vs_beam_25.summary.json
```

Use `--push-timeout-seconds` and `--beam-timeout-seconds` only when the two
solvers should have different per-board budgets. `--timeout-seconds` sets both.

The Beam/CNN timeout is a benchmark cap, not the service default. The production
wrapper's default `ATTEMPT_TIMEOUT_SECONDS` is 300 seconds; this comparison uses
30 seconds per Beam/CNN phase to match the existing push hard-tail benchmark.

| Solver | Verified solved | Accuracy |
|---|---:|---:|
| Push V2.9 | 326 / 341 | 95.60% |
| Beam/CNN production path | 226 / 341 | 66.28% |
| Push V2.9 + Beam/CNN fallback | 329 / 341 | 96.48% |

Beam/CNN-only wins:

- `20260219 Untangled`
- `20260701 Intersection`
- `20260823 Spill the Beans`

Both failed:

- `20250927 Do Not Pass`
- `20251212 -_-`
- `20251228 Cornered`
- `20260124 In the Goal`
- `20260125 Harmonic Progression`
- `20260208 Toric Lens`
- `20260314 Tee Off`
- `20260502 Hexagon`
- `20260523 Anchor`
- `20260524 Gears`
- `20260614 Yellow Light`
- `20260705 High Speed`

Push-only wins: 103 boards. The full row-level comparison is checked in at
`debug-artifacts/push_vs_beam_full.csv`; the most strategically important
push-only wins are the former hard-tail boards `20250928`, `20251005`,
`20251116`, `20251207`, `20251221`, `20260220`, `20260301`, `20260627`, and
`20260802`.

Conclusion: Beam/CNN is no longer competitive as the primary path under this
budget, but it remains valuable as a narrow fallback because it recovers three
boards the push solver misses.

## Next Steps

1. Stabilize the 12 combined failures by replaying oracle/known-solution data
   where available and separating true no-known-solution cases from search
   misses.
2. Build a push-state training set that includes failed frontier states, not
   only solution-path siblings; the current state-action hints are useful but
   mostly memorized.
3. Prototype FESS-style feature buckets for the 12 combined failures before
   adding heavier ML dependencies.
4. If the feature buckets plateau, train a compact policy/value model for
   push-level states and use it only inside verified search.
5. Re-run `benchmark_push_vs_beam` after each hard-tail change and track
   push-only, Beam-only, and combined-failed sets as first-class metrics.

## Executive Summary

V2 should keep the push-level Sokoban solver as the correctness core and add a
portfolio search layer around it. The current failures do not point to a broken
state model. They point to hard-tail search control: the correct next push is
usually ranked near the top locally, but global weighted A* still spends too
much time in broad X-clearing plateaus.

The recommended V2 architecture is:

1. Keep `solver/push_solver/core.py` as the verified push engine.
2. Add a portfolio controller that runs several search schedules against the
   same successor generator and verifier.
3. Add macro successors for common forced push corridors and X-clearing moves.
4. Add stronger, provably safe deadlock and relevance checks.
5. Add a learned push ranker only as an ordering prior, not as the source of
   truth.
6. Always verify emitted keystrokes with the independent replayer.

The key design choice: ML should rank push-level successors and choose restart
schedules. It should not replace legality, state transitions, goal tests, or
verification.

## Current Evidence

V1 has become a strong baseline:

- Abdullah's latest reported broader board-corpus test puts standalone
  `push_solver` accuracy at **93.8%**.
- The reported combined `push_solver` plus `beam_search_and_cnn` approach
  reaches **98%** total board accuracy.
- On the original 20-board historical benchmark, the known hard failures are
  `20250928 Jailbreak` and `20251005 Strange Fit`.
- On the expanded 100-board benchmark, the current hard-tail failures are:
  `20250928 Jailbreak`, `20251005 Strange Fit`, `20251116 Minefields`,
  `20251207 Thread the Needle`, `20251212 -_-`, `20251213 The Pits`,
  `20251214 Unboxing`, `20251221 Imposter`, and `20251228 Cornered`.
- The known beam/CNN solution file has non-empty known solutions for several of
  those failures, but also has null solutions for some boards that V1 can solve.
  That means the beam/CNN corpus is useful training data, but not a complete
  oracle.
- Oracle-rank diagnostics on boards with known solutions show the next correct
  push is usually locally near the top after the current top-K line-plan work.
  Example median oracle ranks are roughly 1 to 2.5 on the hard boards with
  non-null solutions.

Interpretation:

- The push abstraction is correct and worth keeping.
- The push solver is now strong enough to be the default first solver in a
  portfolio, assuming the 93.8% result is reproduced by a checked-in benchmark.
- Beam/CNN still matters because it appears to cover different failures and
  raises combined coverage to 98%.
- Local successor ordering is no longer the only problem.
- The remaining problem is heavy-tail search: long phases where many X pushes
  look equally plausible before any O-distance score improves.
- V2 needs reproducible portfolio benchmarking, rank-discrepancy search, macro
  moves, and learned/global ordering rather than more single-queue hand-tuning.

## V2 Goals

1. Reproduce and check in the benchmark configuration for the reported 93.8%
   standalone push accuracy and 98% combined push plus beam/CNN accuracy.
2. Solve every board in the expanded 100-board benchmark within the production
   time budget, or clearly classify unsolved boards by missing-known-solution
   status and search exhaustion.
3. Beat the current beam/CNN system on solved count first, then on median time.
4. Preserve deterministic, verifier-backed solutions.
5. Keep easy-board speed close to V1. V2 should not slow down boards that V1
   already solves in a few milliseconds.
6. Produce reusable training and debugging artifacts so the hard tail gets
   smaller over time.

## Non-Goals

- Do not promise keystroke optimality in fast mode.
- Do not introduce unsafe pruning just to pass the current benchmark.
- Do not accept ML-generated move strings without replay verification.
- Do not build a raw board CNN as the primary solver. A CNN can be useful later,
  but the first learned component should be a push-level ranker because it can
  consume the same features the classical search already understands.

## Proposed Package Layout

Keep the current public API compatible:

```text
solver/push_solver/
  core.py                 # current push engine and baseline weighted A*
  verify.py               # independent keystroke verifier
  bench.py                # benchmark CLI
  debug_oracle_rank.py    # promote to supported diagnostic tool
  portfolio.py            # V2 portfolio controller
  search_strategies.py    # individual queue/search implementations
  macro_successors.py     # derived push macros
  deadlock_patterns.py    # safe deadlock and relevance tables
  policy_features.py      # feature extraction for push candidates
  rank_policy.py          # ranker interface and exported model inference
  training_export.py      # generate supervised ranking datasets
  benchmark_hard_tail.py  # fixed hard-tail regression suite
```

`core.py` should stay small enough to trust. V2 code should wrap it instead of
turning it into one giant solver file.

## Core Interfaces

### SearchResult

All strategies should return the same shape, so the portfolio can race or
sequence them.

```python
@dataclass(frozen=True)
class SearchResult:
    solved: bool
    moves: str | None
    final_board: tuple[tuple[str, ...], ...] | None
    pushes: tuple[Push, ...]
    nodes_expanded: int
    peak_closed_size: int
    elapsed_ms: float
    failure_reason: str | None
    strategy: str
```

`PushSolveResult` can either be extended or adapted into this type.

### Strategy

```python
class SearchStrategy(Protocol):
    name: str

    def solve(
        self,
        board,
        *,
        max_nodes: int | None,
        timeout_seconds: float | None,
        policy: PushRankPolicy | None = None,
    ) -> SearchResult:
        ...
```

Strategies must share:

- `parse_board`
- `reachable`
- `successors`
- `goal_info`
- `reconstruct_moves`
- `verify`

Strategies may differ in priority calculation, restart behavior, beam width, and
whether macro successors are enabled.

### PushRankPolicy

```python
class PushRankPolicy(Protocol):
    name: str

    def score(
        self,
        *,
        parent: State,
        push: Push,
        child: State,
        features: Mapping[str, float],
    ) -> float:
        """Higher means the push should be explored earlier."""
```

The first implementation should be `HandRankPolicy`, which reproduces current
V1 biases through the new interface. The learned implementation can then be
swapped in without changing search code.

## Search Portfolio

V2 should not depend on one queue formula. Hard Sokoban-like puzzles are
notoriously heavy-tailed, and different priority orders can vary by orders of
magnitude.

Start with a sequential portfolio, because it is simple and deterministic:

```text
total budget = timeout_seconds

1. V1 weighted A*, current settings, small budget
2. More greedy weighted A*, lower g influence, small budget
3. Rank-discrepancy search, medium budget
4. Macro-enabled weighted A*, medium budget
5. Policy-guided weighted A*, remaining budget
```

Later, the same strategy list can be run in parallel workers with a shared stop
flag. The first verified solution wins.

Recommended initial schedule for a 15s production cap:

```text
0.00s - 0.75s   V1 baseline, catches easy boards
0.75s - 2.00s   greedy V1 variant, catches shallow non-optimal paths
2.00s - 5.00s   rank-discrepancy fallback
5.00s - 10.00s  macro-enabled weighted A*
10.00s - 15.00s policy-guided weighted A* or widened discrepancy search
```

The portfolio should report every attempted strategy in debug mode:

```json
{
  "board_id": "20251005",
  "solved": true,
  "winner": "policy_weighted_astar",
  "attempts": [
    {"strategy": "v1", "elapsed_ms": 750, "nodes": 8300, "result": "timeout"},
    {"strategy": "rank_discrepancy", "elapsed_ms": 1910, "nodes": 2871, "result": "solved"}
  ]
}
```

## Search Strategies

### 1. Baseline Weighted A*

This is the current solver. Keep it as the first portfolio entry because it is
fast on easy boards and already verified.

Priority:

```text
priority = g + weight * h + hand_bias
```

### 2. Greedy Weighted A*

Use a lower `g` coefficient to push through plateaus where the direct route is
temporarily longer in push count.

Priority:

```text
priority = 0.25 * g + weight * h + hand_bias
```

This strategy is not optimal. It is only for fast mode.

### 3. Rank-Discrepancy Search

The oracle diagnostics suggest known-solution pushes are usually near the top
locally. Rank-discrepancy search exploits that by minimizing the sum of local
successor ranks instead of raw heuristic priority.

For each expanded state, rank successors using the policy score. The best child
has rank 0, next rank 1, and so on.

Priority:

```text
priority = discrepancy + 0.10 * g + 0.25 * h
discrepancy(child) = discrepancy(parent) + local_rank(child)
```

This should be implemented as a complete best-first strategy with a node cap,
not as recursive DFS. It already showed promise on at least one hard-tail board
in prototype testing.

### 4. Policy-Guided Heuristic Search

Use a learned or hand-built ranker to add a path prior.

```text
priority = g_weight * g + h_weight * h - policy_weight * policy_log_prior
```

Where:

```text
policy_log_prior(child_path) =
    policy_log_prior(parent_path) + log_softmax(policy_score(child among siblings))
```

This keeps the model as an ordering signal while preserving exact legality and
verified reconstruction.

### 5. Macro-Enabled Weighted A*

Macro successors add derived actions alongside normal one-push successors. A
macro must still be expandable into concrete pushes and later into keystrokes.

Macros should never hide illegal intermediate states. Each internal push must
pass:

- destination not occupied
- X-loss check
- deadlock check
- state normalization

Useful first macros:

- `push_until_blocked`: when a piece can only continue through a one-wide
  corridor and every intermediate push remains legal.
- `clear_target_x`: push an X off a current top-plan player target if the clear
  direction is forced.
- `advance_assigned_o`: follow the canonical reverse-distance route for an O
  while each next stand cell is reachable and no blocker intervention is needed.

Macro actions should be optional per strategy. Do not enable them in baseline
V1 until they have their own tests.

## Macro Successor Rules

A macro candidate is valid only if:

1. It expands to a finite tuple of existing `Push` objects.
2. Replaying the tuple from the parent state produces the advertised child
   state.
3. Every intermediate state is legal.
4. The macro's cost equals the number of pushes in the tuple.
5. The macro is not the only way to reach that child. Normal one-push successors
   remain available for completeness.

Recommended type:

```python
@dataclass(frozen=True)
class MacroPush:
    name: str
    pushes: tuple[Push, ...]
    child: State
    child_region: frozenset[int]
    cost: int
```

Search strategies that consume macros must update parent reconstruction with the
full push tuple, not the macro object alone.

## Safe Deadlock And Relevance Layer

Deadlock pruning is where V2 can gain the most, but it is also where it can
become unsound. Every pruning rule must have a short proof or must be demoted to
an ordering penalty.

Safe candidates:

### Static O Dead Cells

Already implemented. Keep it.

An O on a floor cell that cannot reach any win-line cell under wall-only reverse
push reachability can never contribute to a solution.

### Pair Impossibility Table

Precompute, for every unordered pair of O cells, whether there exists any win
line and assignment that both O cells can still reach under wall-only
push-distance maps.

If no assignment exists, prune.

This is safe because it ignores X blockers and player access, so it
over-approximates what is possible. If even the over-approximation says no, the
state is dead.

### Sealed Component Target Check

If both O pieces are in positions that can only complete lines whose player
target lies in a sealed component unreachable by the player under all movable
piece placements, prune.

This is harder to prove. Implement first as a penalty or diagnostic. Promote to
pruning only after a proof and exhaustive small-board tests.

### X Loss Predecessor Avoidance

Current code rejects any X push that creates an immediate X line. V2 can add a
soft penalty for pushes that create two Xs on many lines with an open third
cell. This should be ordering-only, not pruning.

### Relevance Cuts

Relevance cuts from Sokoban are useful but risky. In V2, implement them only as
a priority penalty:

- Penalize pushes far from all top-K line plans.
- Penalize X shuffling that neither expands the player region nor clears a
  target/route/stand cell.
- Reward pushes that affect multiple top-K plans.

Do not hard-prune "irrelevant" pushes in fast mode until the hard-tail suite has
regression tests proving no solved board is lost.

## Learned Push Ranker

The learned component should answer one question:

> Given a state and its legal push successors, which successor should be tried
> first?

This is a ranking problem, not an image classification problem.

### Why A Push Ranker Before A CNN

- It uses exact legal successors, so it cannot suggest illegal moves.
- It is board-size agnostic.
- It can be trained from known push sequences and from solver-generated data.
- It is debuggable: feature importances and per-push scores can be printed.
- It works with weighted A*, discrepancy search, and beam search.

### Training Sources

1. Known beam/CNN solutions that replay successfully.
2. V1/V2 solutions generated by any strategy.
3. Hard-tail manual or oracle solutions as they are discovered.
4. Negative siblings from the same parent state.

Do not train on move strings blindly. First replay each solution through the
push engine and convert it into a push sequence. Discard or quarantine any row
that fails replay.

### Dataset Rows

For each parent state on a known solution path:

- Generate all legal successors.
- Label the successor matching the next known push as positive.
- Label all siblings as negative.
- Store board id, depth, parent hash, push, features, and label.

Useful JSONL shape:

```json
{
  "board_id": "20251005",
  "depth": 14,
  "parent_key": "...",
  "push": {"piece": "X", "cell": 37, "move": "L"},
  "label": 1,
  "sibling_count": 9,
  "features": {
    "piece_is_o": 0,
    "h_delta": -1.5,
    "region_delta": 4,
    "legal_o_push_delta": 2,
    "clears_player_target": 1,
    "enters_route_cell": 0
  }
}
```

### Feature Set V1

Start with scalar features. They are cheap and match the current code.

State-level:

- push depth `g`
- current heuristic `h`
- number of legal O pushes
- player reachable region size
- count of Xs on any candidate line
- count of top-K line plans
- best-plan score gap between rank 1 and rank K

Push-level:

- piece type: O or X
- move direction one-hot
- source cell row/col normalized by board size
- destination cell row/col normalized by board size
- source/destination degree in floor graph
- whether source/destination is on a top-K line
- whether source/destination is the player target of a top-K plan
- whether source/destination is in a top-K O route
- whether source/destination is in a top-K stand-cell set
- whether X push clears a target, line, route, or stand cell
- whether X push enters a target, line, route, or stand cell
- O assigned-target push-distance before and after
- `h_delta`
- player-region size delta
- legal-O-push-count delta
- X-line-threat delta
- number of top-K plans improved
- number of top-K plans harmed

Sibling-level:

- local rank under current hand bias
- normalized hand-bias score
- difference from best sibling score
- sibling count

### Model Progression

Use the simplest model that improves the hard-tail suite.

1. Export current hand features and use a fixed linear score. This proves the
   interface.
2. Train logistic regression or pairwise linear ranking offline. Export weights
   as JSON for dependency-free inference.
3. If linear ranking plateaus, try a small gradient-boosted tree model offline.
   Export to a compact JSON evaluator or keep it behind an optional dependency.
4. Only then consider a neural model. If a neural model is used, prefer a small
   MLP over scalar features before a raw CNN.

Raw CNN is a fallback, not the next best step.

## Training And Evaluation Commands

Suggested commands:

```bash
python3 -m solver.push_solver.training_export \
  --solutions solver/gymnasium_register/all_boards_heuristic_cnn_solutions.jsonl \
  --out data/push_ranker/train.jsonl

python3 -m solver.push_solver.train_ranker \
  --train data/push_ranker/train.jsonl \
  --out solver/push_solver/models/push_ranker_linear.json

python3 -m solver.push_solver.benchmark_hard_tail \
  --strategy portfolio \
  --timeout 15
```

The trainer can be added after the export format is stable. The benchmark should
exist first so every change can be judged against the same hard-tail list.

## Benchmark Plan

Create three benchmark tiers.

### Tier 1: Unit And Invariant Tests

Run on every change:

```bash
python3 -m unittest tests.test_push_solver -q
```

Required invariants:

- every successor is replay-valid
- every reconstructed move string verifies
- no X-loss state is accepted
- player-region normalization is stable
- macro expansion equals repeated one-push expansion
- deadlock pruning never removes known solved fixture paths

### Tier 2: Historical 20

This is the easy compatibility benchmark. V2 should solve all 20 within the
current cap.

### Tier 3: Expanded Hard Tail

Fixed board ids:

```text
20250928 Jailbreak
20251005 Strange Fit
20251116 Minefields
20251207 Thread the Needle
20251212 -_-
20251213 The Pits
20251214 Unboxing
20251221 Imposter
20251228 Cornered
```

Track:

- solved count
- verified count
- elapsed milliseconds
- nodes expanded
- peak closed size
- winning strategy
- push length
- move-string length
- whether the board has a non-null known beam/CNN solution

### Tier 4: Full Corpus

Run against the full JSONL board corpus and report three numbers:

- solved by V2
- solved by beam/CNN known data
- solved by both

The target is not just to solve the current nine failures. The target is to make
future failures diagnosable.

## Implementation Phases

### Phase 0: Stabilize Diagnostics

Deliverables:

- Promote `debug_oracle_rank.py` from scratch tool to supported diagnostic.
- Add a hard-tail board list fixture.
- Add a benchmark mode that prints per-strategy attempts.
- Add a solution replay converter that turns move strings into push sequences.

Exit criteria:

- Existing unit tests pass.
- The hard-tail benchmark reproduces the current failure list.
- Oracle-rank diagnostics work for every board with a non-null known solution.

### Phase 1: Portfolio Controller Without ML

Deliverables:

- `portfolio.py`
- `search_strategies.py`
- Baseline weighted A* strategy
- Greedy weighted A* strategy
- Shared result type and attempt logging

Exit criteria:

- Portfolio result matches baseline on easy boards.
- No strategy bypasses verifier-backed reconstruction.
- Easy-board median time does not regress materially.

### Phase 2: Rank-Discrepancy Strategy

Deliverables:

- Best-first rank-discrepancy implementation.
- Shared local successor ranking function.
- Per-depth rank/discrepancy metrics.

Exit criteria:

- Solves at least one current hard-tail board that baseline misses, or gives a
  clear diagnostic reason why it does not.
- Does not reduce solved count on the historical 20.

### Phase 3: Macro Successors

Deliverables:

- `macro_successors.py`
- `MacroPush` type
- corridor macro
- target-X clear macro
- assigned-O advance macro
- macro expansion tests

Exit criteria:

- Macro replay equivalence tests pass.
- At least one hard-tail board shows fewer expanded nodes or lower elapsed time.

### Phase 4: Safe Deadlock Tables

Deliverables:

- O-pair impossibility precompute.
- Deadlock regression tests using all known solution paths.
- Ordering-only relevance penalties for non-proven rules.

Exit criteria:

- No known solved path is pruned.
- Hard-tail node counts improve or stay neutral.

### Phase 5: Ranker Data Export

Deliverables:

- `policy_features.py`
- `training_export.py`
- JSONL dataset writer
- replay validation for all imported solution strings
- feature dumps for individual states

Exit criteria:

- Generates labeled examples from all non-null known solutions.
- Produces no examples from invalid or unverifiable solution rows.
- Feature generation is deterministic.

### Phase 6: Learned Linear Ranker

Deliverables:

- `rank_policy.py`
- dependency-free JSON linear model inference
- optional offline trainer
- policy-guided weighted A*
- policy-guided rank-discrepancy search

Exit criteria:

- Learned ranker improves at least one hard-tail board without reducing total
  solved count.
- Model inference cost is small relative to successor generation.
- Benchmark output includes model name and version.

### Phase 7: Heavy-Tail Scheduling

Deliverables:

- Restart schedules by strategy.
- Optional per-board timeout predictor from cheap static features.
- Optional parallel portfolio runner.

Exit criteria:

- Full-corpus solved count improves over Phase 6.
- Production timeout behavior is predictable and observable.

## Acceptance Criteria For V2

V2 is ready to replace the current production default when:

1. `python3 -m unittest tests.test_push_solver -q` passes.
2. Every returned solution verifies by replay.
3. The original 20-board benchmark is 20/20 under the production timeout.
4. The checked-in benchmark reproduces or improves the reported 93.8%
   standalone push accuracy.
5. The hard-tail benchmark reports which strategy won each solved board.
6. Full-corpus comparison against beam/CNN known data is generated and the
   combined portfolio reproduces or improves the reported 98% accuracy.
7. There is a documented fallback path for every remaining unsolved board:
   timeout, node cap, exhausted, missing known solution, or suspected invalid
   board data.

The aspirational target is 100/100 on the expanded benchmark. The engineering
target is a solver whose failures are classified and reproducible, not opaque.

## Risks And Mitigations

### Unsafe Pruning

Risk: a deadlock or relevance rule silently removes valid solutions.

Mitigation: every hard prune needs a proof and tests against known solution
paths. Unproven rules are ordering penalties only.

### Overfitting To Current Hard Boards

Risk: hand-tuned constants solve the current list but fail future boards.

Mitigation: use portfolio strategies and learned ranking features that explain
general push usefulness. Keep a full-corpus benchmark.

### ML Dependency Creep

Risk: production solver becomes hard to install and debug.

Mitigation: make learned inference optional and dependency-free at first. Export
linear weights as JSON. Keep hand policy as fallback.

### Easy-Board Regression

Risk: V2 overhead slows boards V1 already solves immediately.

Mitigation: portfolio always runs baseline first with a small budget. Feature
extraction and ML scoring are only used after baseline misses.

### False Confidence From Beam/CNN Data

Risk: known solution file has nulls and may contain invalid or incomplete rows.

Mitigation: replay every imported solution. Treat nulls as unknown, not
unsolvable.

## References

These are useful background sources for the implementation direction:

- Sokoban PSPACE-completeness:
  https://sokoban.dk/wp-content/uploads/2016/02/Sokoban-is-PSPACE-complete.pdf
- Rolling Stone Sokoban solver and domain-dependent enhancements:
  https://webdocs.cs.ualberta.ca/~jonathan/publications/ai_publications/ijcai99.pdf
- Sokoban deadlocks, relevance, and search enhancements:
  https://svn.sable.mcgill.ca/sable/courses/COMP763/oldpapers/junghanns-01-sokoban.pdf
- Relevance cuts for Sokoban:
  https://webdocs.cs.ualberta.ca/~jonathan/publications/ai_publications/rc.pdf
- Pattern databases for Sokoban:
  https://www.sciencedirect.com/science/article/pii/S0004370215000867
- Policy-guided heuristic search:
  https://arxiv.org/abs/2103.11505
- Learning policy guidance for heuristic search:
  https://ojs.aaai.org/index.php/AAAI/article/download/17469/17276
- Policy/value learning for Sokoban:
  https://www.cs.cornell.edu/gomes/pdf/2022_feng_arxiv_sokoban.pdf

## Recommended First PR

Do not start with ML. Start with the scaffolding that makes every later
experiment measurable:

1. Add `portfolio.py` with baseline and greedy weighted A* strategies.
2. Add `benchmark_hard_tail.py` with the fixed hard-tail board ids.
3. Promote `debug_oracle_rank.py` into a supported diagnostic command.
4. Add solution replay-to-push-sequence export.
5. Run the hard-tail benchmark and record a baseline table in the PR.

After that, implement rank-discrepancy search. It is the highest-leverage next
algorithmic step because it directly matches the oracle-rank evidence and does
not require training data.
