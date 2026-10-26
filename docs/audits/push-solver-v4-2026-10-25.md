# Push Solver V4 implementation and release evidence

V4 is experimental. `DAILY_PUSH_SOLVER_VERSION` still defaults to `v3`; direct
and custom solving retain their existing routing. Neither archive replacements
nor a V4 daily cutover is authorized by the evidence below.

## Inputs and reference

The immutable [corpus](../../debug-artifacts/v4-2026-10-25/catalog.jsonl) contains
397 official boards through October 25, with date and catalog SHA-256 identity.
The catalog source receipt is alongside it. Original historical verified paths
and parser-repair paths remain in `debug-artifacts/catalog-audit-2026-10-22/`;
they are separate evidence, not search results for V4.

The fresh sequential V3 baseline independently verified **390/397** boards with
30 seconds / 500,000 nodes, quality search disabled for the coverage comparison.
V3's search, geometry, pruning, optimizer, and verifier code are unchanged from
`82dd036`. The optional diagnostics field does not affect V3 execution. See
`v3-baseline.jsonl` and `v3-baseline.metadata.json` in the new artifact directory.
Some later reference measurements ran alongside independent research processes
on this 16-core host; these are local coverage measurements, not isolated
runtime or deployment-equivalence claims.

V3 failed December 12/28, February 15, March 14, May 24, September 13, and
October 4. This fresh measured reference is distinct from the nine dates with
unsolved production records.

## Implementation

- `solver/push_solver/v4.py`: integer-mask occupancy, reachability, line and route
  intersections; cached reverse-push maps keyed by permanent blockers and target;
  bounded invocation-local caches; lazy O-assignment geometry; complete short
  plan caching; incremental X-loss checks; all O pairs and OOO wins.
- Four generator sessions retain frontier, immutable reconstruction links,
  independent dominance tables, and unfinished successor generation. The default
  allocation is rank-discrepancy, weighted, macro, committed beam at 4:2:1:1.
  A slice yields at 250 milliseconds or 128 expansions. Uniform and weighted-heavy
  alternatives are explicit benchmark options.
- The absolute deadline covers preprocessing, search, reconstruction, replay, and
  optional keystroke improvement. Search stops 250 milliseconds early for the
  final reconstruction/replay window. Quality search uses V3's exact-player
  keystroke objective through cooperative V4 successors and shares the total node
  budget. A verified incumbent survives interrupted or failed quality search.
- Up to three deterministic wall-only routes, at most shortest+2 pushes, inform
  congestion ordering. Primitive pushes remain available in the three unfiltered
  searches. Committed beam has its own relevance/escape band and soft novelty
  penalties; these do not prune another strategy's frontier.
- `solve_v3` is an explicit reference alias. `solve_v4` has matching budget and
  quality arguments; extra experimental options select schedules, individual
  strategies, route count, and an offline feature-only candidate. V4 never calls
  the exact-action hint policy. The replay verifier is unchanged.
- Daily results retain optional internal diagnostics. `daily_solve_runs` commits
  a receipt before parsing/search, a checkpoint after parsing, another immediately
  after push search, and a terminal receipt afterward; no database
  transaction spans network access or solving. Protected solved rows can remain
  solved while a retry's own receipt records unsolved/failed. Terminal receipts
  cannot be overwritten. A worker exiting during the long fallback leaves a
  running receipt containing its already-completed push diagnostics. Push expansions/quality expansions and fallback replay
  transitions have separate counters. The legacy public count remains compatible.
- Authenticated `POST /debug/push-benchmark/{date}` executes push search without
  fallback or persistence. `GET /debug/solve-runs/{date}` reads attempt receipts.
  Search receipts include source/model fingerprints, CPU/wall time, limits,
  per-strategy counters/frontiers, cache measurements, and runtime memory/quotas.

## First milestone and experiments

The initial corrected V4 implementation solved Bubble Wrap locally in 1.75s
(portfolio) / 1.15s (rank alone), versus the earlier unchanged-V3 measurements of
22.2s (portfolio) / 4.7s (rank alone). All paths passed the independent verifier.
These comparisons are local measurements from separate runs.

An initial one-CPU, 512 MiB Linux amd64 container using the repository Dockerfile
completed ten fresh-process and ten warm-process Bubble Wrap runs in roughly
2.6–2.8 seconds each. One V3 run in the same constrained image failed after
32.4 seconds externally measured. These limits have not yet been established as
the deployed service's limits. This experiment does not prove the cause of the
October 25 production failure. Container receipts pin the source used at build
time; later experimental engine changes require new release receipts.

Initial V4 production-target coverage was **4/9**: September 11 and October
17/18/25. The five persistent hard targets still failed at the daily budget.
The initial full-corpus run found losses against V3, including committed-beam
recovery cases. Commitment scoring and novelty handling were subsequently
corrected; both initial and corrected corpus evidence are retained separately.
The corrected local target portfolio solved **5/9**, including Cup of Tea. The
read-only deployed nine-target run solved **4/9**: Cup of Tea exhausted its push
budget there. None of these results meets the release requirement.

Offline sweeps recovered independently verified paths for -_- (32.8s committed
beam), Surrounded (182.1s rank), Peninsula Problems (186.0s weighted), and Cup of
Tea (16.9s rank). A research path is not a daily-budget portfolio result.

The pushed engine also verified Bubble Wrap through the actual deployed API in
2.93s (204 moves, quality disabled). Its source hash matched `a9470d5`. The
subsequent sequential deployed target probe took 4.23s for the same date. The
public API is served by the Vercel Python entry point; the separate hosted Cloud
Run deployment is not proof of the Vercel runtime configuration. That runtime
reported Linux/Python 3.12.14 and no readable cgroup quotas, so the earlier
one-CPU container is a controlled comparison, not a deployment-matched gate.

Local validation passed **158 Python tests**, including real Postgres concurrency,
all 360 small-board arrangements with one wall/X and two O pieces, differential
successors, three-O cases, session resumption, shared budgets, durable progress,
and fail-closed release receipts. Hosted `a9470d5` validation passed the earlier
157 Python tests, all web tests/lint/build, and the Cloud Run deployment. The
checkpoint hardening is validated separately at its own commit.

Offline sweeps use 300 seconds / 5 million nodes per strategy and stop a board
when a verified path appears. These are research limits only. Sweeps for separate
boards may run on separate host cores; they cannot establish production timing.
No saved path is loaded as a search hint by V4.

`train_v4_candidate.py` reuses the existing supervised exporter and pairwise
trainer. Original board identity (including rotations/reflections) determines
splits. The nine targets are development data. The candidate stays in the artifact
directory, has no exact-action keys, and is never selected by the daily pipeline.
Initial fits were rejected after nonfinite legacy-export features produced NaN
coefficients. Their metrics are invalid and their artifacts are quarantined under
`invalid-candidates/`. The corrected exporter sanitizes, bounds, and scales
features; model loading rejects nonfinite coefficients and scoring rejects
nonfinite predictions. The corrected candidate's held-out *action-ranking*
accuracy is 439/1071 (41.0%). This is not held-out search coverage or evidence of
unseen-board generalization. It remains outside the production model path.

## Final cache refinements and candidate status

Macro continuation now computes the requested legal push directly, reusing an
edge cache, instead of constructing unrelated sibling successor lists. Frozen
constraint decisions are cached by O mask, permanent-blocker mask, and the player
component after removing movable pieces. Goal geometry is cached by O positions;
nonterminal plan scores depend on O/X masks, with goal reachability still checked
against the current player component. Route choices are reused within each score
calculation. These are invocation-local caches, not stored solution hints.

This reduced direct beam research solves for -_- from 32.8s to 3.8s and Tee Off
from 42.8s to 6.8s in local intermediate measurements. The final pinned default
4:2:1:1 schedule solved **7/9** mandatory dates locally; Tee Off and Peninsula
Problems exhausted their budgets. Uniform also solved 7/9. Weighted-search-heavy
allocation lost -_- and did not improve coverage. An additional explicit
1:1:1:4 beam-heavy alternative solved **8/9**, still missing Peninsula Problems.
No alternative has been promoted to the daily path.

The final one-CPU/512 MiB Linux beam-heavy probe also failed Peninsula Problems;
Tee Off finished in 28.2s, showing limited runtime margin. This is not the complete
180-run deployment-matched release matrix. The ordinary matrix remains gated on
complete target and corpus coverage.

Local validation now passes **162 Python tests**. The final corpus runs and
hosted/deployed receipts are indexed in the artifact directory; only receipts
matching the final engine fingerprint and exact budget/configuration can satisfy
the gate. Earlier experiments remain audit evidence, not release evidence.

## Reproduction and release gates

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --version v3 --output /tmp/v3.jsonl
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --output /tmp/v4.jsonl --gate-v3 /tmp/v3.jsonl
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --targets --schedule uniform --output /tmp/uniform.jsonl
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --targets --schedule weighted --output /tmp/weighted.jsonl
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --sweep --fresh-process --ids 20251212 --output /tmp/research.jsonl
```

`gate()` fails on missing/duplicate boards, stale engine hashes, mixed configurations,
research budgets or single-strategy receipts, mismatched identities, time/node overruns,
exact-action hints, any lost V3 solution, or an unsolved mandatory target.
`repetition_gate()` requires all 180 target receipts (nine boards × ten fresh
processes and ten warm runs). Container/deployment equivalence and hosted tests
are additional gates, not inferred from a local Python pass.

The complete 180-run matrix, daily selector switch, and archive replacements
remain gated on full target/corpus coverage. No fallback success or longer
research solve can satisfy the daily push gate. Once all gates pass, archive
repairs must preserve prior records and invalidate the affected page caches.
