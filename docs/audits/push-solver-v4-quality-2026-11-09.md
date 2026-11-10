# V4 move quality — November 9, 2026

The new candidate retains **397/397 verified solutions** and all nine required
targets while reducing mean keystrokes from **66.20 to 39.17 (40.84%)**. Both
complete current-engine native cohorts pass the original V3 no-coverage-loss
gate. Every result stays within the same **30-second / 500,000-expansion** limit.

| Complete native V4 cohort | Verified | Mean moves | Median moves | Total moves |
| --- | ---: | ---: | ---: | ---: |
| Previous coverage control, quality disabled | 397/397 | 66.20 | 57 | 26,282 |
| Rewrites plus walking-cost search, 10s quality window | 397/397 | 44.05 | 32 | 17,488 |
| Rewrites, vicinity, and walking-cost search, 10s window | 397/397 | 40.63 | 31 | 16,132 |
| Same final engine, 30s quality window | **397/397** | **39.17** | **31** | **15,549** |

The quality window ends at solve-start plus 10 or 30 seconds, capped by the
same overall deadline. It shares the expansion counter with coverage search.
The 30-second setting permits optimization of solutions found after the first
ten seconds. It retains a verified incumbent through interruption and errors.

These are sequential native development-corpus runs, with independent experiments
overlapping on the host. They establish measured coverage and path quality on
the frozen 397 boards. Deployment throughput and repeated-run acceptance remain
separate. Daily selection still defaults to V3; no production solution records
or models were replaced by this work.

The final 10-second-window cohort peaks at 25.03s and 181.9 MiB process RSS;
the 30-second-window cohort peaks at 26.05s and 198.0 MiB. Memory figures are
process high-water marks, not per-board allocations. No timed V4 solve exceeds
30 seconds. On the earlier 397 control paths, the final 10-second candidate is
shorter on 360 boards and equal on 37; the 30-second candidate is shorter on
370, equal on 26, and longer on one. Every invocation nevertheless retains a
path no longer than its own first verified incumbent.

## Matching operational V3 comparison

The fresh operational V3 quality-enabled run solves 383/397; the earlier
quality-disabled release reference solved 390/397. The V4 coverage gate still
uses that original 390-board reference and requires all nine mandatory targets.
The fresh run's lower coverage does not relax the gate. Different runtime
conditions and overlapping experiments limit interpretation of timing differences.

Both fresh V3 and the matching V4 run use a 30-second total limit, 500,000
expansions, and a 10-second quality window. On their **383 identical verified
board identities**:

| Matching comparison | V3 | V4 |
| --- | ---: | ---: |
| Mean moves | 46.42 | **38.02** |
| Total moves | 17,780 | **14,561** |

That is **18.10% fewer moves**. V4 is shorter on 158 of these boards, equal on
191, and longer on 34. This paired comparison is distinct from V4's full
397-board average. The 30-second quality-window candidate averages 37.63 on
the same subset, but uses a different quality allocation and is reported
separately.

## Research and selected approaches

The optimization objective is total direction keystrokes, including walking.
Sokoban research often measures minimum pushes instead, as explicitly described
by [Pereira et al., IJCAI 2016](https://ijcai.org/proceedings/2016/100).
The distinction matters here: the same pushes can require very different walks.

[Tim Wheeler's push-permutation experiments](https://timallanwheeler.com/blog/2022/02/20/push-permutation-optimization-of-sokoban-solutions/)
motivate changing the order of independent work and grouping a piece's pushes.
The implemented rewriter also removes unnecessary operations, blocks, and entire
pieces' work when a complete replay still wins. Piece identities track relative
push instructions during each candidate replay, so moving a block of operations
does not assume its old source coordinates. The evaluator recomputes shortest
walks, accepts an earlier reachable win, and rejects every intermediate X triple.

[JSoko's optimizer documentation](https://jsokoapplet.sourceforge.io/help/help/optimizier.html)
describes restricting improvement search to configurations near an incumbent
solution. Our bounded neighborhood includes the current invocation's push
layouts and configurations one push away. Dijkstra search over that graph keeps
the exact player cell and prices an edge as shortest walk plus one push. It can
find different push directions and winning lines beyond simple permutations.
The layout pool is capped at 8,192; if the incumbent itself exceeds that size,
the neighborhood phase returns without changing it.

[Stern et al., Potential-Based Bounded-Cost Search and Anytime Non-Parametric A*](https://goldberg.berkeley.edu/pubs/AI-Journal-Anytime-Nonparametric-A-Star.pdf)
informs the incumbent-and-bound approach. This implementation keeps the existing
weighted search as a further repair phase. The hand heuristic includes congestion
and remains nonadmissible; published optimality/suboptimality bounds are not
claimed. Strict improvement and independent replay are the acceptance criteria.

## Implementation and ablations

The first prototype operated only on saved control outputs for diagnostic
post-processing. Runtime code consumes the solution produced by the current
solver call, with no archive, date, board hash, model, or saved answer lookup.

- Local deletion/reordering shortened 291 control paths, from 26,282 to 19,868
  moves, in about two seconds across the stored outputs. Increasing its allowance
  from 100ms to 500ms produced exactly the same paths and totals.
- The one-push neighborhood shortened another 244 paths, taking that offline
  total from 19,868 to 17,784 moves. The complete probe took about 2.8 seconds.
- The first live integrated candidate, without the neighborhood phase, completed
  397/397 at 44.05 mean moves. It remains a distinct archived control.
- The final engine composes 100ms-bounded rewrites, a 200ms-bounded neighborhood,
  and weighted walking-cost repair. The first two phases ordinarily finish far
  before their caps. The weighted phase stops after one second without an
  improvement, including when its initial incumbent is already good.

Walking distances are computed once per expanded quality state and shared by
its outgoing edges. Straight push macros usually need no additional walking
between their pushes. Terminal pricing chooses the nearest reachable winning
target across all winning lines, rather than the first enumerated line.

Every neighborhood expansion and weighted-search expansion debits the same
solver node budget. Linear rewrite replay evaluations have separate counters.
Every accepted candidate is independently replayed; the last verified solution
survives expiry, node exhaustion, or a later optimization error. The original
coverage schedule and legal primitive successors are retained.

The benchmark CLI records `--quality-seconds`, including fresh-process children,
and rejects resumed source/configuration changes. Corpus and repetition gates
reject mixed quality windows. The engine fingerprint now includes
`path_optimizer.py`. The authenticated read-only benchmark endpoint accepts
`quality_timeout_seconds`; its default remains zero.

## Verification and runtime limits

Python validation passes **199 tests with four optional Postgres skips**. New
fixtures cover removal of unnecessary X work, nearest-target selection, OOO
wins, intermediate X losses, interruption after a verified improvement, a case
shortened from 11 to 3 moves by changing a push direction, and shared quality
node accounting. Targeted lint and `git diff --check` pass.

The separately captured official November 7–8 boards also pass the final
30-second-window candidate: Windmill shortens from its published 77 moves to
44 in a local 3.93s probe, and Mind The Gap from 91 to 46 in 2.07s. These are
additional read-only tests outside the frozen corpus; published records were
not changed.

`summary.json` independently replays every successful path and checks its final
board and complete push trace, verifies monotonic improvement against each
invocation's first incumbent, and checks quality expansion accounting. The V3
comparison uses a fresh full operational V3 run with the matching 10-second
quality window and the same total budgets. Per-board comparisons use the same
official identities; a V3 solved-subset mean is not compared directly with V4's
full 397-board mean.

The selected local Linux ARM64 probe uses one CPU, 512 MiB, no network, and a
recorded dependency image with current source mounted read-only. It recovers
Immovable Objects with 94 moves in 24.97s and Cold Feet with 74 moves in 15.01s.
Peninsula Problems still times out at about 29.77s in that environment. The full
deployment-matched repetition matrix and deployed candidate probes remain open,
so the default daily selector is unchanged.

Receipts, source snapshots for the controls, per-board results, runtime versions,
and reproducible aggregates are in
[`debug-artifacts/v4-quality-2026-11-09/`](../../debug-artifacts/v4-quality-2026-11-09/).

## Reproduce

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --schedule coverage --quality-seconds 30 \
  --output /tmp/v4-quality-30.jsonl \
  --gate-v3 debug-artifacts/v4-2026-10-25/v3-baseline.jsonl
```

Use `--quality-seconds 10` for the matched V3 quality-window comparison.
For a deployed read-only check after deploying this code, use
`POST /debug/push-benchmark/YYYY-MM-DD` with existing cron authorization and
`{"version":"v4","schedule":"coverage","quality_timeout_seconds":30}`.
