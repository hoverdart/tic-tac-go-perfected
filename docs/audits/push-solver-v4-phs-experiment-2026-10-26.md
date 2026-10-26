# V4 policy-guided search experiment — October 26

Implemented in `04d5cf5`, with trained linear and small neural push policies,
PHSh, Levin search, PHS*, explicit portfolio controls, and reproducible data.
The experiment finds useful new search behavior, but **does not pass release**.
The current best native portfolio remains beam8/straight at 395/397. The final
adaptive learned portfolio solves 393/397 and all nine production targets,
recovering Cold Feet while losing other V3 boards. Both deployed PHS* probes
still time out. Daily selection remains at its V3 default; no archive records
were replaced and no production model or selector was promoted.

## What was implemented

The search follows the evaluation functions in
[Orseau and Lelis, Policy-Guided Heuristic Search with Guarantees](https://ojs.aaai.org/index.php/AAAI/article/view/17469/17276),
including the more aggressive PHS* variant from section 3.2. With
`g = push_depth + 1` and `L = -log(path_probability)`, the implemented log
priorities are:

- Levin: `log(g) + L`.
- PHSh: `log(g + h) + L`.
- PHS*: `log(g + h) + (1 + h/g) * L`.

Each path also retains the maximum priority of its ancestors. Here `h` is V4's
existing hand heuristic, not a learned value function. PHSh/Levin keep separate
Pareto labels for depth, negative log probability, and ancestor priority. A
shorter but less probable path cannot automatically erase a longer, more
probable one. PHS* uses first-expanded-state graph closure and only same-depth
queue dominance: the depth-dependent exponent makes the PHSh depth rule unsafe.
These graph rules, nonadmissible hand heuristic, and finite budgets do **not**
inherit the paper's tree-search guarantees.

Every lane uses V4's unchanged legal primitive successors and the existing
independent replay verifier. No macros enter policy probabilities. A five-percent
uniform mixture gives every retained legal primitive action probability at least
`0.05 / action_count`. Path probability stays in log space. Search frontiers,
parents and labels survive the existing 250-ms / 128-expansion quanta, debiting
the same absolute deadline and 500,000-node counter as the other lanes.

The new policy extractor computes 33 bounded numeric features from occupancy,
reachability, O mobility, hand scores, route/standing-cell/target congestion,
local piece adjacency, and X threats. It loads no board identity, saved action,
or trajectory at runtime. Inference is Markov in V4 state, with an invocation-local
bounded probability cache. Training and inference use the same extractor.
The strict model schemas reject action hints, unknown fields, nonfinite weights,
invalid shapes and invalid probability controls.

There are two model forms: a 33-weight conditional softmax, and a single hidden
layer of 32 tanh units with 1,120 parameters. Linear inference uses Python;
neural inference uses NumPy. Model hashes identify both independently. Candidate
files stay outside the production model directory. Only an explicit `policy_path`
or an authenticated benchmark request selecting a named candidate enables them.

## Data and fitting

The corpus is the frozen 397 official boards through October 25. For each board,
training selects the verified path with the fewest pushes, then fewest keystrokes,
from the prior verified-path archive, V3 reference and round-two V4 corpus.
An absent demonstrated primitive causes export to fail rather than silently
skipping it. Export stops once V4 can already walk to a win.

The result is **5,176 decisions and 46,188 legal action examples**. Ninety-nine
forced decisions are excluded from fitting and probability metrics. Canonical
original board identity keeps rotations, reflections and duplicates together:

| Partition | Board entries | Used for |
| --- | ---: | --- |
| Train | 310 | Coefficients |
| Development | 13 | Coefficients; nine targets and four known regressions |
| Calibration | 38 | Temperature only |
| Evaluation | 36 | Reported metrics only |

All 323 train-plus-development entries contribute a non-forced decision.
The split is identity-based; no evaluation identity contributes a coefficient
or temperature gradient. However, this entire corpus has been used in previous
solver benchmarking, so these metrics establish no unseen-board generalization.

Fitting uses board-balanced demonstration cross-entropy, a uniform probability
mixture, regularization, deterministic shuffling and Adam. Linear fitting uses
80 epochs, neural fitting 120, seed `20261026`. Temperature is selected by
board-averaged demonstration NLL on calibration identities, separately from the
36 evaluation identities. Multiple legal alternatives can lead to solutions;
these labels identify one demonstrated action, not all winning actions.

| Evaluation metric | Linear | Neural |
| --- | ---: | ---: |
| Demonstrated-action top-1 accuracy | 66.5% | 65.4% |
| Temperature selected on calibration data | 1.1487 | 1.2599 |
| Board-mean NLL before temperature fitting | 0.9354 | 0.9595 |
| Board-mean NLL after temperature fitting | 0.9609 | 0.9685 |

Temperature fitting improves its own calibration partition, but worsens held-out
NLL. Neural ranking accuracy is also slightly lower despite useful search
recoveries. Neither accuracy nor calibration-set improvement is a release gate.
An explicit temperature-one neural ablation solves Cold Feet in 6.63s but fails
Peninsula; the fitted-temperature neural model solves both. No board-specific
choice between temperatures is used.

The entire dataset was re-exported after the runtime code was finalized. Its
four arrays and compressed file hash match exactly. The input/model manifests,
original and reproduced provenance, and equality receipt are committed.
Native training/inference used NumPy 2.5.1; the local Linux image has NumPy 2.5.4.
The deployed dependency version was not independently measured. That is another
reason the local container is not a deployment-matched acceptance run.

## Search results

All native corpus comparisons use 30 seconds / 500,000 expansions, quality budget
zero, no exact action hints, no fallback, and independent replay verification.
Each corpus runs sequentially; independent experiments ran concurrently on the
native Mac. Intermediate fingerprints remain separate from the final engine.

| Configuration | Corpus | Nine targets | V3 coverage losses |
| --- | ---: | ---: | ---: |
| Fresh beam8/straight control, final engine | **395/397** | **9/9** | 2 |
| PHSh + linear, beam-heavy portfolio, intermediate | 394/397 | 8/9 | 2 |
| PHS* + neural, beam-heavy portfolio, intermediate | 394/397 | 8/9 | 2 |
| PHS* + neural, policy-heavy portfolio, intermediate | 393/397 | 8/9 | 3 |
| PHS* + neural, adaptive portfolio, final engine | 393/397 | **9/9** | 4 |

The beam-heavy policy schedule weights rank/weighted/macro/beam/policy as
1:1:1:8:2; the policy-heavy schedule uses 1:1:1:2:8. These are weighted visits
under the existing time/expansion quanta, not guaranteed percentages of CPU time.
A development hypothesis uses the ordinary beam8 schedule when there are at most
16 feasible initial goal commitments, otherwise the policy-heavy schedule. It
caches this initial pool and records the chosen schedule. It uses no dates or
board hashes. This adaptive hypothesis does not improve aggregate coverage and
is not promoted or further tuned to route individual failed dates.

The adaptive portfolio loses November 16, February 19, March 1 and August 23
against V3. The control loses March 1 and August 2. In particular, recovering
Cold Feet does not erase the newly introduced losses. A union of different
configurations is not a single-budget result.

Standalone PHSh with the linear policy solves 0/9 targets. The four-board Levin
probe solves 0/4. Standalone neural PHS* solves 6/9 targets; it misses -_-,
Surrounded and Tee Off. It provides two useful independently verified recoveries:

| Board | Beam8 control | Standalone neural PHS* | Pushes | Moves |
| --- | ---: | ---: | ---: | ---: |
| Cold Feet, August 2 | Timeout | **8.85s** | 16 | 112 |
| Peninsula, September 13 | 26.23s | **14.86s** | 23 | 149 |

A uniform-policy PHS* control fails both boards. Three fresh-process and three
warm-process runs per recovered board pass **12/12**. All finish within the daily
budget; fresh process durations are below 15.11s including startup, and
warm solves are below 15.67s. These are selected development-case repetitions,
not the required 180-case acceptance matrix.

Local Linux ARM64, one CPU / 512 MiB, solves Cold Feet in 10.11s and Peninsula in
16.88s. Image identity, limits and process memory receipts are stored separately.
The final native control's process peak RSS is 171.2 MiB; adaptive is 251.8 MiB.
These are process high-water marks, not per-board allocations. On shared V3
successes, the control uses 24,566 moves versus V3's 23,632 across 388 boards;
adaptive uses 23,410 versus V3's 23,566 across 386 boards. The sample sets differ:
coverage, memory, speed and move quality must be assessed separately.

## Deployed evidence and negative experiments

Authenticated read-only Vercel probes match all final runtime helper hashes and
the neural model hash. Both fail at about 29.8s: Cold Feet expands 2,819 states;
Peninsula expands 4,034. Native successful runs expand 2,908 and 6,860 respectively.
Deployed thread CPU time is about 29.6s and 29.4s. Cold preprocessing is 152ms;
warm Peninsula preprocessing is 2.5ms. The runtime gap remains after initialization.
These facts do not establish CPU throttling or exact cross-platform frontier
correspondence. No daily solve, fallback, archive write or ledger write is called.

A separate isolated prototype accepts a generated winning child immediately,
rather than waiting for its PHS queue removal. It saves **one expansion** on
each recovered board. That is not an explanation for the deployed timeouts;
the prototype is retained as negative evidence and is not in the runtime path.

Offline Immovable Objects probes use 300 seconds / five million nodes. Committed
beam finds a verified 77-push / 327-move solution in 204.0s. This is longer than
the existing 33-push demonstration, so it is not selected for training. Neural
PHS* remains unsolved at 300s after 171,206 expansions. Neither result satisfies
a daily release gate and the daily budget was not widened.

After both model artifacts were frozen, October 26's official “Left Or Right”
was tested outside the 397-board corpus. Beam8, standalone neural PHS*, and the
adaptive portfolio all solve it in 21–35ms. This one current-day probe is not a
held-out test set and is not added to fitting or calibration.

## Verification and decision

Local validation passes **185 Python tests**, including isolated Postgres
integration, probability normalization/floors, malformed models, cache isolation,
training/inference equality, data partitioning, Pareto tradeoffs, resumability,
shared budgets, and independent small-board exhaustive oracles. Web tests
pass 34/34; lint and targeted Python checks pass.
[Hosted CI](https://github.com/hoverdart/tic-tac-go-perfected/actions/runs/40288169463)
passes Python/Postgres tests, web tests/lint and build for `04d5cf5`.
[Cloud Run deployment](https://github.com/hoverdart/tic-tac-go-perfected/actions/runs/40288169574)
also succeeds; its workflow result is separate from a solver runtime benchmark.

The benchmark engine fingerprint now covers PHS and policy runtime helpers, not
only `v4.py`. Resuming a changed policy or configuration is rejected. Repetition
gates reject duplicate or extra records. Earlier experiments retain their own
fingerprints and are not valid current-engine release receipts.

**Decision: retain the implementation as an explicit experiment.** No tested
portfolio improves the 395-board control, the V3 no-regression gate is unmet,
and both deployed recovery probes fail. Full deployment-matched ten-fresh /
ten-warm runs and archive replacements remain unaccepted. The next measured
bottlenecks are deployed scoring throughput and selecting useful goal commitments;
a goal-conditioned policy is a better next learning hypothesis than simply
increasing this model's size or routing dates to their successful configurations.

Reproduce the learned search:

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --ids 20260802 20260913 --strategy phs_star \
  --policy debug-artifacts/v4-phs-2026-10-26/policy-mlp.json \
  --output /tmp/phs-recoveries.jsonl
```

Artifacts and reproduction commands:
[`debug-artifacts/v4-phs-2026-10-26/`](../../debug-artifacts/v4-phs-2026-10-26/README.md).
