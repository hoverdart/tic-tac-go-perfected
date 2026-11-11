# Optimized-path solver

## Objective and boundary

Minimize total direction keystrokes: walking and pushing each cost one. A
solution with more pushes may be preferable. This is a standalone, offline
optimizer seeded with independently verified solutions. It does not replace
V4, change daily/custom routing, read the production database, or publish paths.

V4 remains the coverage solver. V1–V4 and older verified solutions provide
alternative trajectories. The optimizer retains the shortest valid input and
accepts only independently verified, strictly shorter replacements.

## First implementation

The new `solver.optimized_paths` package has three modes:

1. **graph**: replay all seeds into a directed graph of exact states and observed
   single-keypress edges. BFS can combine trajectory segments only when O
   positions, X positions, and the actual player cell all match. This is optimal
   only within the observed graph, not the full puzzle.
2. **repair**: start with the graph result, reuse V4's bounded push rewriter, then
   search a bounded neighborhood of archived piece layouts. Search prices each
   push as its shortest preceding walk plus one. Neighborhood exhaustion never
   implies global optimality.
3. **exact**: start with the verified graph/rewriter result, then A* over every
   legal primitive push with exact walking costs. This mode uses no beam,
   relevance filter, learned pruning, or unproved deadlock rule. State reopening
   handles improved arrival costs. Final walks into winning lines are charged.

Every mode has a shared elapsed-time/expansion budget and keeps its verified
incumbent on interruption. Seed intake is verified before use, including when
the search budget is zero. Preprocessing has separate counters and shares the
wall-clock deadline. The budget is cooperative, not a hard real-time guarantee.

### Two-O lower bound

For a board with exactly two movable Os, remove Xs but retain walls, both Os,
the player, and legal push rules. A multi-source reverse BFS from all relaxed
winning states computes exact remaining keypress distances. Every real move
maps to a legal relaxed move, so this table never overestimates real distance.
The table is a pattern database; it is used only after construction completes.
A state/memory cap or interruption discards the partial table and falls back
to zero. Boards with more Os also use zero initially, preserving OOO wins.

For 394/397 frozen boards there are exactly two movable Os. Their theoretical
relaxed placement counts have median 19,635 and maximum 124,992. These counts
include unreachable placements and are not construction timings.

### Proof semantics

`optimality_proven` means a verified incumbent meets the global lower bound,
or the complete, unfiltered A* frontier cannot contain a shorter path.
The bound includes a partially expanded node if execution is interrupted.
Restricted repair/graph exhaustion is never a proof. A zero-length verified
solution is trivially optimal. V4's existing scores are not admissible lower
bounds and must not be used for certification.

The push graph is sufficient for the move objective: between two consecutive
pushes the pieces stay fixed, so replacing the intervening walk with a BFS
shortest walk cannot worsen the solution. A terminal walk to the closest
reachable winning cell covers solutions that finish without another push.
The exact player cell belongs to the dominance key. Immutable parent links
prevent reopening a state from corrupting a previously queued trajectory.

The global search prunes only when `g + h >= incumbent_length`. On interruption
its lower bound is the minimum unresolved frontier value, including any node
whose successors were only partially generated, capped by the incumbent length.
These are algorithmic optimality claims, not independently checkable exhaustive
proof logs; independent BFS fixtures test the kernel and relaxed distances.

## Bank integrity and evaluation

The benchmark accepts explicit catalog and archive manifests. Each source is
fingerprinted. Archives must match the catalog's exact board hash; dates alone
are insufficient. Every distinct candidate is replayed independently. Input
schemas and rejected rows are recorded. Invalid paths and transformed/mismatched
boards are not silently accepted. Missing seeds remain visible in results.

Compare output against the **shortest input per board**, not just V4. Archive
reuse must be reported separately from fresh solving and future-board learning.
Keep all cases in each cohort, including timeouts, missing seeds, and failures.
Record mode, limits, input/source hashes, moves, lower bound, proof status,
termination, time, expansions, and phase diagnostics.

The preceding read-only probe inspected 11 archive files and replay-verified
1,357 distinct paths: best-input mean 35.05038, observed-graph mean 34.97733,
397/397 coverage. This is a reproduction target, not an optimality claim. The
qualified V4 hosted mean of 40.46851 is a separate fresh-solver measurement.

## First complete development results

All three final cohorts use the same 11 source archives, 397 board identities,
1,357 distinct replay-verified input paths, one-second nominal wall budget,
and 100,000 shared search-expansion cap per board. They ran sequentially on
native ARM64 macOS / Python 3.12.13. These are development results, not a hosted
qualification or an isolated CPU timing comparison. Some lightweight tests ran
alongside the cohorts. Time-limited search results can vary between runs.

| Cohort | Verified | Mean moves | Moves saved vs best inputs | Boards shortened | Globally proved shortest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best input per board | 397/397 | 35.05038 | — | — | Not assessed |
| Observed graph | 397/397 | 34.97733 | 29 | 6 | 0 |
| Radius-one repair | 397/397 | **33.61713** | **569** | **75** | 0 |
| Exact A* + relaxed database | 397/397 | 33.68766 | 541 | 74 | **282** |

Repair reduces moves by **4.09% relative to the shortest archived inputs**.
Exact mode certifies 282 paths as globally shortest; the remaining 115 retain
verified best-known paths and explicit lower bounds. The two modes are separate
cohorts, not a per-board union presented as one budget. Exact mode favors a
complete search and proofs; restricted repair can find improvements faster.

All **1,191** final output receipts independently replay, match their final
boards and source/input fingerprints, retain all expected case identities,
respect the expansion cap, and pass non-regression and claim-accounting checks.
There are no errors or missing seeds. The largest measured times are 8.55 ms
(graph), 316.38 ms (repair), and 1,051.91 ms (exact). The last includes cooperative
interruption/finalization overhead and must not be advertised as a hard 1s cap.

Engine fingerprint:
`0f5b0af3233e495265f1abd423f92992d98c726a6284df5dbd761f7778dd79a3`.
Receipts, source/input manifests, summaries, and independent verification reports
are in [`debug-artifacts/optimized-paths-2026-11-10`](../debug-artifacts/optimized-paths-2026-11-10).
The original qualified V4 source files remain unchanged.

Twenty-seven new tests cover input integrity, trajectory joins, geometry,
pattern distances, interruption bounds, OOO wins, CLI output, and receipt
tampering. Together with related existing solver tests, **107 tests pass**.
The optimality tests use an independent legacy unit-step BFS oracle. An additional
corpus check selects the twelve shortest certified paths, breaking ties by ID,
and independently confirms all twelve minimum lengths; its 50,000-node per-case
cap is recorded in `exact-1s/independent-oracle.json`. This does not independently
recheck all 282 proofs.

An earlier pre-formatting development cohort was retained locally in
`/tmp/tictacgo-paths-initial-20bffa13`; the table above uses only the final
fingerprinted cohorts checked into this branch.

## Extended correctness testing

The original implementation passes CI at `d639cdc`: **242 Python tests** pass
against PostgreSQL, the same 242 pass in the compiled production container,
and all **34 web tests**, lint, and the isolated web build pass.

Two independently generated batches cover **1,200 boards** across 3x3, 3x4,
4x3, 4x4, and 2x4 geometries, with two/three movable Os, walls, and up to three
Xs. The independent legacy unit-step BFS finds 573 solvable cases, exhausts
619 unsolvable cases, and rejects eight initial X losses. Of the solvable cases,
294 deliberately supply a longer-than-optimal seed. The optimizer passes:

- 2,292 checks at four different expansion caps;
- 424 rotation/reflection checks;
- 7,632 deterministic checkpoint interruptions during exact search.

Every returned interval contains the independent minimum; every completed proof
agrees with it. The relaxed starting bound also never exceeds the oracle cost.
All 282 original corpus certificates were attempted with independent unit-step
BFS: **115 are independently confirmed**, and 167 hit the 20,000-node oracle cap
and remain **inconclusive**, with no contradictions. Evidence is in `stress/`
and `extended/stress-xs3/`. The original harness snapshot is retained with its
source fingerprint. These are correctness probes, not timing measurements.

### Receipt configuration regression

Testing found that v1 could accept one case copied from a different mode when
its counts happened to match. A concrete reproduction replaces exact-mode
case `20250927` with its repair-mode receipt. A separate regression changing
the recorded time budget also failed to raise before the fix.

Schema **v2** binds the engine, complete archive inputs, options, and recorded
runtime in a configuration digest included in metadata, **every case**, and the
summary. Tests reject both changed metadata and a row with another configuration
even after the metadata/summary digest is updated. All **109 related tests**
pass after the fix. This is consistency checking, not a cryptographic signature.
The optimization/geometry/lower-bound kernels are unchanged by this repair.

The original v1 cohorts remain preserved at `d639cdc` with their original source
fingerprints. The current verifier intentionally requires v2; use a separate
worktree at `d639cdc` to replay v1 evidence with its matching verifier. New
cohorts under `extended/` use v2 and a frozen source snapshot in `run-plan.json`.

## Longer budgets and independent proof checks

The v2 runs use engine fingerprint
`5e12fd2eb16f0e3c2ff9582c4ace02e2138fe2fecf24b1496e9e9e6703179555`.
All five complete cohorts below retain **397/397** verified paths with no errors
or missing seeds and a 100,000-expansion cap. They run sequentially against the
same original bank; elapsed budgets are cooperative. Process peak RSS comes
from macOS `/usr/bin/time -l`, not an isolated hosted measurement.

| Cohort | Nominal seconds | Mean moves | Paths shortened vs inputs | Globally shortest | Process peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Observed graph | 1 | 34.97733 | 6 | 0 | 49.98 MiB |
| Radius-one repair | 1 | 33.61713 | 75 | 0 | 51.91 MiB |
| Exact A* | 1 | 33.83627 | 73 | 281 | 126.48 MiB |
| Exact A* | 10 | **32.93955** | **97** | **324** | 777.72 MiB |
| Radius-two repair, 32,768 layouts | 10 | 33.09068 | 93 | 0 | 65.58 MiB |

The ten-second exact run shortens 35 paths relative to its matching one-second
control, saving another 356 moves; none grow longer. Widening repair shortens
45 paths relative to radius one, saving 209 moves; none grow longer. The earlier
v1 one-second result differs slightly under its timer, so it remains a separate
cohort instead of replacing the new control. No failed runs are omitted.

Of the 73 unresolved exact-ten-second cases, 72 hit the node cap and one hits
the time limit. More wall time alone therefore has limited value. A published
development selection tests the three smallest positive unresolved gaps and
the unresolved additional-O case at 60 seconds / 200,000 expansions:

| Guard | Best path | Lower bound | Outcome |
| --- | ---: | ---: | --- |
| Unraveled | 37 | 37 | Globally shortest |
| House of Cards | 39 | 39 | Globally shortest |
| Lost At Sea | 37 | 37 | Globally shortest |
| Imposter | 44 | 31 | Node cap; remains unproved |

All four verify. Three new proofs bring the completed-search total **across
runs** to 327. This guard subset is not a 397-board result. Its largest elapsed
time is 7.27s and process RSS is 268.06 MiB. The ten-second full cohort reaches
10.48s including interruption/finalization; it is not a hard 10s service limit.

### Independent checker

`scripts/check_optimized_paths_independently.py` implements its own set-based
push transitions and wall-only single-O matching bound. It shares neither the
optimizer's bitmask transitions nor its pattern database. Its tests include
deliberately nonoptimal paths and independent BFS-generated puzzles, so a valid
long path cannot pass merely by replaying successfully.

At one second / 20,000 nodes per claim, this checker independently confirms
**203 of the full cohort's 324 proofs**. The other 121 checks are inconclusive
(64 node caps, 57 timeouts). All three selected new guard proofs are inconclusive
under this smaller oracle budget. The combined independent confirmations remain
203 unique cases, including the earlier unit-step confirmations. There are
**no contradictions**; inconclusive results are never counted as confirmations.
All **116 related Python tests** pass after adding the second oracle's fixtures.

### Consolidated bank and optimal substructure

`inputs-refined.json` explicitly combines the original eleven archives with nine
completed optimizer cohorts. Every distinct input is replayed again. The best
verified path per board now averages **32.05793 moves** (12,727 total), an
**8.54% reduction** from the original bank's 13,915 moves. This is cumulative
archive aggregation across trials and budgets, **not one fresh solve budget**.
Observed-graph search finds no further shortening in this consolidated bank.

`scripts/analyze_optimized_path_experiments.py` validates all v2 receipts and
checks every stronger lower bound against the consolidated best paths. All
327 completed proofs agree across runs. No additional proof arises solely from
matching an unfinished bound to another run's path. The consolidated certified
paths provide **9,021 exact suffix states** by optimal substructure. Checking
86,919 archived suffix positions, including 14,675 matching certified states,
finds no shorter suffix or conflicting distance label.

The analysis and `certified-paths.jsonl` are in `extended/analysis/`; the latter
retains per-board path and lower-bound provenance for future training export.
Whole-board identity splits are still required before learning. No model was
trained or enabled during these tests, and production V4 remains untouched.

## Next milestones

The first implementation reproduces the graph baseline, improves beyond the
best inputs, and cross-checks its exact mode/lower bounds against independent
unit-step BFS. Completed measurements are recorded above.

- Evaluate longer runs with explicit memory/state bounds: the current node
  cap, rather than wall time, limits most remaining cases. The full ten-second
  and development-selected sixty-second controls above are complete.
- Measure additional optimality proofs and the unresolved lower-bound gaps,
  alongside mean moves and memory.
- Add persistent checkpoints, selective larger repairs, and additional-O
  abstractions after the first implementation is validated.
- Later train keystroke-aware ranking/value guidance. Archived suffix lengths
  are feasible upper bounds, not exact labels. Hold out whole board identities
  and their transformations. Learned estimates must not certify pruning.
- Globally optimal trajectories permit exact remaining-distance labels on
  their suffix states, by optimal substructure. Keep those labels distinct
  from unproven feasible suffix lengths. They still do not make a learned
  predictor admissible on unseen states.
- Production integration comes later: V4 publishes first; a separate worker
  improves verified seeds and conditionally replaces only a strictly longer
  solution for the identical board. Preserve source/provenance and receipts.

## Running the prototype

From the repository root, a single-board input is JSON containing `board` in
the usual U/O/X/B/empty-cell format and `seeds`, an array of move strings. For
example, `{"board":[["U","O","","O"]],"seeds":["R"]}`.

```sh
PYTHONPATH=. .venv/bin/python -m solver.optimized_paths \
  --input /tmp/board-and-seeds.json --mode exact --seconds 10

PYTHONPATH=. .venv/bin/python -m solver.optimized_paths.benchmark \
  --manifest debug-artifacts/optimized-paths-2026-11-10/inputs.json \
  --output-dir /tmp/optimized-paths-graph --mode graph --seconds 1

PYTHONPATH=. .venv/bin/python -m solver.optimized_paths.benchmark \
  --manifest debug-artifacts/optimized-paths-2026-11-10/inputs.json \
  --output-dir /tmp/optimized-paths-repair --mode repair --seconds 1

PYTHONPATH=. .venv/bin/python -m solver.optimized_paths.benchmark \
  --manifest debug-artifacts/optimized-paths-2026-11-10/inputs.json \
  --output-dir /tmp/optimized-paths-exact --mode exact --seconds 1

PYTHONPATH=. .venv/bin/python -m unittest tests.test_optimized_paths

PYTHONPATH=. .venv/bin/python -m solver.optimized_paths.verify_benchmark \
  /tmp/optimized-paths-exact \
  --manifest debug-artifacts/optimized-paths-2026-11-10/inputs.json

PYTHONPATH=. .venv/bin/python scripts/stress_optimized_paths.py \
  --output-dir /tmp/optimized-paths-stress --generated 600 --max-xs 3 \
  --cohort /tmp/optimized-paths-exact \
  --manifest debug-artifacts/optimized-paths-2026-11-10/inputs.json

PYTHONPATH=. .venv/bin/python scripts/check_optimized_paths_independently.py \
  --cohort /tmp/optimized-paths-exact \
  --manifest debug-artifacts/optimized-paths-2026-11-10/inputs.json \
  --output-dir /tmp/optimized-paths-independent --seconds 1 --max-nodes 20000

PYTHONPATH=. .venv/bin/python scripts/analyze_optimized_path_experiments.py \
  --output-dir /tmp/optimized-paths-analysis
```

Output directories must be empty. Benchmark output includes an immutable input
manifest, source fingerprints, all case receipts, and a summary. Use `--ids`
with one or more catalog IDs for a subset, and `--radius` / `--max-layouts` to
bound the repair neighborhood. Subset runs explicitly retain their selected IDs.
The separate receipt verifier replays every saved answer, checks final boards,
non-regression, source/input identity, case completeness, accounting, and proof
flag/bound consistency. It does not independently repeat the full optimal search.
No command contacts Google, the API, or the production database.

## Research basis

- [Push permutation optimization](https://timallanwheeler.com/blog/2022/02/20/push-permutation-optimization-of-sokoban-solutions/):
  reorder pushes to reduce walking.
- [JSoko optimizer](https://jsokoapplet.sourceforge.io/help/help/optimizier.html):
  multiple seed solutions, vicinity and interval search.
- [Experience Graphs](https://publications.ri.cmu.edu/e-graphs-bootstrapping-planning-with-experience-graphs):
  reuse previous planning trajectories.
- [Optimal Sokoban heuristics](https://ijcai.org/proceedings/2016/100):
  admissible pattern databases; its push objective differs from our move cost.
- [ARA*](https://proceedings.neurips.cc/paper/2003/hash/ee8fe9093fbbb687bef15a38facc44d2-Abstract.html)
  and [ANA*](https://goldberg.berkeley.edu/ana/): future anytime-search candidates.
  The first exact implementation is ordinary incumbent-bounded A*, not either
  of these algorithms.
