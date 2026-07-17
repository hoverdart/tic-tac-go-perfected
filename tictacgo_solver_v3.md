# Tic Tac Go Push Solver V3

## Why V2 Was Slow And Verbose

The July 17, 2026 `Cylinder` production solve exposed two independent issues:

- The independent verifier called `parse_board()` after almost every move.
  `parse_board()` builds reverse-push routes and every reachable O-pair table,
  even though replay only needs walls, legal moves, wins, and X losses. A profile
  of Cylinder showed 69 parses and 6.6 seconds in verification versus 5.8 seconds
  across the search strategies themselves. Production returned 88 moves in
  48.01 seconds despite reporting only 329 searched states.
- V2 minimizes pushes, not keystrokes. It normalizes the player to one
  representative of the reachable region, which is correct for board coverage
  but discards the exact position needed to price walking. On the 223 corpus
  boards solved by both push and Beam/CNN, V2 returned more moves on 181; its
  median move-count ratio was 1.52x and its median gap was 13 keystrokes.
- The portfolio returned its first verified solution. A later strategy could be
  both faster and shorter on the same board, but it never ran after an early V2
  incumbent succeeded.

## V3 Design

V3 is a coverage-preserving anytime layer, not a replacement for V2's legality
or search coverage:

1. `solve_v2()` receives the complete caller-provided node and time budgets.
2. A verified V2 solution becomes an incumbent that V3 never discards.
3. If the board was solved early, V3 uses the remaining portion of a 10-second
   quality window. Unsolved boards keep the full 30-second coverage search.
4. Quality search uses `(normalized push state, actual player cell)` as its key,
   charges shortest walking distance plus pushes on every edge, and uses the V2
   heuristic/macros for ordering.
5. Branch-and-bound only accepts a strictly shorter reconstructed path. The
   independent verifier remains the final acceptance gate; timeout, node cap,
   error, or invalid output returns the original V2 solution.

The verifier now normalizes once, precomputes valid three-cell lines once,
tracks the player during replay, checks X loss only after an X push, and checks
the final win without constructing any search tables.

## Coverage Rule

Board coverage must never decrease in V3 or later push-solver versions. This is
enforced in two ways:

- Runtime invariant: quality search only runs after V2 has solved and verified
  the board, and all failure paths return that incumbent.
- Release gate: run the full benchmark and use
  `python3 -m solver.push_solver.coverage_guard BASELINE.csv CANDIDATE.csv`.
  The command fails if even one previously verified board is missing.

`solve()` is now V3; `solve_v2()` and `solve_v1()` remain callable for regression
and research comparisons. Production and backfill records use `push-v3`.

## Reference Results

Local focused runs after implementation:

| Board | Previous production | Local V2 incumbent | Local V3 |
|---|---:|---:|---:|
| 2026-07-16 Hourglass | 42 moves / 2.28 s | 42 moves / 0.48 s | 19 moves / 2.29 s in the full run |
| 2026-07-17 Cylinder | 88 moves / 48.01 s | 124 moves / 1.23 s | 34 moves / 4.58 s in the full run |

Timing varies by machine and strategy deadline, so releases are evaluated by
verified coverage, keystroke count, and wall time together.

The completed 341-board run is checked in at
`debug-artifacts/push_solver_v3_full.csv`:

- verified coverage: **337/341 (98.83%)**, up from the stronger checked-in V2
  baseline of 327/341;
- coverage regressions: **0** against both prior full-corpus CSVs;
- quality improvements: **208/337** solved boards;
- total keystrokes: **20,480 → 15,683**, saving **4,797** moves;
- remaining failures: `20251212`, `20251228`, `20260314`, and `20260524`.

The six-worker pass produced contention timeouts for two known solved boards;
both were rerun in isolation and passed. The final CSV contains those verified
isolated rows, which is the release procedure for any apparent parallel-run
regression.

### Validation Commands

```bash
python3 -m unittest tests.test_push_solver tests.test_push_solver_coverage \
  tests.test_solver_service tests.test_daily_solve tests.test_backfill_solutions

python3 -m solver.gymnasium_register.benchmark_push_vs_beam \
  --solver push --workers 6 --push-timeout-seconds 30 \
  --push-max-nodes 500000 --output /tmp/push_solver_v3.csv \
  --summary-output /tmp/push_solver_v3.summary.json

python3 -m solver.push_solver.coverage_guard \
  debug-artifacts/push_solver_v3_full.csv /tmp/push_solver_candidate.csv
```
