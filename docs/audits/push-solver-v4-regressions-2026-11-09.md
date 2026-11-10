# V4 coverage regression work — November 9, 2026

The best prior V4 beam8/straight control solved 395/397 frozen boards but lost
two operational V3 successes: **Immovable Objects (March 1)** and **Cold Feet
(August 2)**. Both V4 attempts timed out at 30 seconds. The V3 reference solved
them in 16.24s and 11.61s respectively. This is the coverage regression discussed
here; deployed throughput and move quality are separate acceptance dimensions.

The final native candidate solves **397/397**, including **all nine mandatory
targets**, with **zero V3 coverage losses**. All 397 paths pass independent
replay against the frozen official layouts. This passes the corpus coverage
gate, not the full release gate.

| Configuration | Verified corpus | Mandatory targets | Lost V3 successes |
| --- | ---: | ---: | ---: |
| Operational V3 reference | 390/397 | 4/9 | — |
| Prior best V4 beam8 control | 395/397 | 9/9 | 2 |
| New V4 coverage candidate | **397/397** | **9/9** | **0** |

In that sequential native cohort, Immovable Objects takes **22.68s**, Cold Feet
**13.65s**, and Peninsula Problems **26.98s**. Every solve stays within the
30-second / 500,000-expansion budget; the largest wall time is 26.98s. Native
process peak RSS is 168.5 MiB. These are development-corpus measurements on the
local host, not unseen-board generalization or deployment timing guarantees.

The new explicit `schedule="coverage"` candidate retains several search
objectives rather than replacing the existing V4 configuration. Daily selection
still defaults to V3. No model, stored solution, or production selector is changed
by this work.

## Why a replacement scorer was insufficient

The previous committed beam picked a line/player target but used the cheapest
line anywhere on the board as its main heuristic. Moving toward another line
could improve that heuristic while moving away from the commitment. A new
commitment score evaluates both O assignments against the selected line and
player target, including route congestion and occupied target penalties. It
reassigns the O pieces as they move; original piece coordinates are not action
hints. A win on any line still has zero cost.

Using this score everywhere recovered both original failures, but introduced
timeouts on Thread the Needle, Imposter, and Unraveled. Simple score averages,
line-only scoring, and changing only the narrow exploration lane did not retain
both sets of successes. Those negative probes are preserved in the artifact
directory. They are diagnostic experiments, not release receipts: in-memory
prototype patches are not represented by their base-file engine hashes.

Unraveled exposed a second allocation problem. Its successful global search
needed a 96-wide beam on an early plan, after the old schedule had exhausted
many plans at width 48. Starting every search at width 96 recovered it quickly
but lost other narrower-beam successes. The candidate retains both widths.

## Implementation

The new coverage beam has independent resumable global-score, commitment-score,
and wider global-score searches. Their visit weights are 2:2:1, with 32-expansion
or 25ms cooperative slices. The wider search starts at width 96. Existing rank,
weighted, and macro searches remain present; the outer coverage allocation is
1:1:1:32. All lanes share the same absolute deadline and 500,000-expansion cap,
including preprocessing, reconstruction, and independent replay verification.
They share successor/geometry caches, not frontier or dominance tables. No
existing saved path, exact action hint, learned policy, CNN fallback, or V3
fallback participates in this candidate.

The extra searches need faster shared scoring:

- Reachability uses bounded grid bit shifts with explicit row-edge masks.
- Wall-only canonical routes are cached per invocation.
- Canonical geometry costs use NumPy bit counts and stable score/index ordering
  on bounded 64-cell pools. Tiny pools, larger boards, and pools above 4,096 rows
  retain cooperative Python scoring. The three dependency manifests now require
  NumPy 2.0 or newer.

The frozen 62-state scoring fixtures still match exactly, including route
alternatives and ties. New tests compare reachability with independent flood
fills on rectangular boards, distinguish global improvement from progress on a
committed line, accept wins outside that commitment, and verify that all three
coverage frontiers share one node budget without exceeding parent slices.

The benchmark CLI and authenticated read-only endpoint accept the new schedule.
Existing schedule names and the daily selector retain their meanings.

## Evidence and remaining limits

Current-engine corpus, focused, Linux, recent-board, and test receipts are under
[`debug-artifacts/v4-regressions-2026-11-09/`](../../debug-artifacts/v4-regressions-2026-11-09/).
`summary.json` provides the complete corpus gate, V3 coverage differences, path
verification, timing, move counts, and memory measurements.

Move quality remains a separate regression: on the 390 shared V3 successes,
total keystrokes increase from 23,773 to 25,062 (**5.42%**), and the median moves
from 53 to 56. Quality optimization is disabled for both sides of this coverage
comparison. No shortest-path or move-quality improvement is claimed.

An intermediate 1:1:1:16 corpus recovered both V3 regressions and retained every
V3 success, but missed mandatory Peninsula Problems: 396/397 and 8/9 targets.
The final candidate increases beam allocation while retaining the other lanes;
the total time and node budgets are unchanged. The intermediate receipts have
`-16` suffixes and must not be merged into the final cohort.

The selected local Linux ARM64 probe uses one CPU, 512 MiB, no network, and a
read-only source mount with a recorded runtime image. It recovers Immovable
Objects in 25.02s and Cold Feet in 14.65s. Peninsula Problems still times out
there at about 29.77s. This is controlled local evidence, not deployment-matched
acceptance.

One fresh-process native Immovable Objects attempt also times out at 29.76s,
with 20.55s of recorded thread CPU time. An intermediate focused native Cold
Feet attempt timed out with 18.62s of CPU work in 29.76s of wall time. These
wall/CPU gaps do not establish a cause; both failed receipts remain part of the
evidence. The mandatory deployment-matched
ten-fresh/ten-warm matrix and deployed candidate probes have not been accepted.
Recovering coverage is therefore insufficient to promote V4 as the daily default.

Python validation runs 189 tests successfully with four optional Postgres
integration skips; targeted lint and endpoint tests pass. No frontend source or
database schema changes are involved.

Reproduce the corpus comparison from the repository root:

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --schedule coverage \
  --output /tmp/v4-coverage.jsonl \
  --gate-v3 debug-artifacts/v4-2026-10-25/v3-baseline.jsonl
```

For the deployed read-only probe after this implementation is deployed, use
`POST /debug/push-benchmark/YYYY-MM-DD` with the existing cron authorization and
`{"version":"v4","schedule":"coverage"}`. This endpoint does not publish
solutions or change the daily selector.
