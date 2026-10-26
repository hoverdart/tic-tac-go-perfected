# V4 round two: Google research and controlled search changes

The strongest native candidate is now **395/397, including all nine production
targets**, using `schedule="beam8"`, `macro_mode="straight"`, and interleaved
commitments. This is an improvement over the prior best V4 result of 392/397
and 8/9 targets. It is still experimental: Immovable Objects (March 1) and Cold
Feet (August 2) remain V3 regressions, and Peninsula Problems fails the deployed
Vercel probe. Daily selection remains V3. Implementation: `be10287`.

## Evidence boundaries

Google's deployed Tic-Tac-Go application and DeepMind's published Sokoban work
are separate evidence sources. No source inspected establishes that the daily
game uses DeepMind's neural agents or policy-tree-search algorithms.

The official [game](https://dailygames.discover.google.com/games/tic-tac-go)
loads a public GoogleGamesPlatformUi module. The inspected `m=Aeygpd` response
was 673,027 bytes, SHA-256
`128674ad4e2d6c62b550e0f05bb58a78fd1983b82e14a0ac6aef3e447bca0b21`,
matching the catalog source used for the frozen October 25 corpus. Its embedded
editor exports board JSON and the flat position string. The editor includes
“Solver Position” and “Pass this to the solver to generate a solution.” This
supports an authoring workflow with a separate solver; it does not identify its
algorithm, implementation language, or whether it uses machine learning. Searches
of the inspected module and public Google sources did not locate that solver's
implementation. Third-party clones and our own project pages are not evidence
of Google's solver internals. No future catalog layouts were added to training
or evaluation.

## Relevant primary research

1. [DeepMind's Boxoban levels](https://github.com/google-deepmind/boxoban-levels)
   supply separate train/validation/test data and harder puzzle sets. The games
   use four boxes and fixed targets. Tic-Tac-Go instead has relational winning
   lines and losing X triples, so their rules, deadlocks, and trained policies
   cannot be imported unchanged. The applicable idea is to expand training with
   verified, generated puzzles while keeping original puzzle identities apart.
2. [Single-Agent Policy Tree Search With Guarantees](https://papers.neurips.cc/paper_files/paper/2018/hash/52c5189391854c93e8a0e1326e56c14f-Abstract.html)
   (Orseau et al., 2018) studies policy-guided enumeration and sampling on Sokoban.
   It makes search effort depend on probability assigned to solution paths.
   This motivates retaining exploration outside the most highly ranked actions.
3. [Policy-Guided Heuristic Search with Guarantees](https://ojs.aaai.org/index.php/AAAI/article/view/17469)
   (Orseau and Lelis, 2021) combines policy and heuristic guidance and evaluates
   it on Sokoban and other deterministic puzzles. A future PHS/Levin candidate
   would need calibrated action probabilities and its own dominance rules;
   raw linear ranking scores are not probabilities. V4's finite beams and time
   caps do not inherit these papers' theoretical guarantees.
4. [An Investigation of Model-Free Planning](https://proceedings.mlr.press/v97/guez19a.html)
   (Guez et al., 2019) studies recurrent networks that exhibit planning behavior.
   It establishes an alternative learning direction, not evidence that a small
   model trained on our 397 boards would have comparable performance. For this
   iteration deterministic legality and independent replay remain authoritative.
5. DeepMind's [Agents that imagine and plan](https://deepmind.google/blog/agents-that-imagine-and-plan/)
   describes learning how much simulation to spend and when to continue versus
   restart trajectories. Its Sokoban experiments used procedural levels and
   limited real attempts. This is further support for investigating adaptive
   search allocation, rather than adopting a large learned world model when we
   already have inexpensive, exact game transitions.

## Implementation response

The existing bitmask engine is retained. Nonterminal scoring caches ranked rows
and materializes only the requested prefix: one best child plan, eight parent
plans, and larger pools only for commitments. Static O-assignment costs and
unique route/line masks are compiled once. The second sort handles only the
refined top 24 candidates. Frozen reference fixtures check exact scores and plan
ordering with one and three routes before and after this optimization.

Goal commitments now own resumable frontiers, cost tables, novelty state, and
phase counters. A focused lane retains the established beam policy while a
narrower lane explores diverse lines, player targets, and O assignments. Pool
widening waits for a minimum amount of work on active goals. Commitments proven
impossible by permanent blockers are excluded from this pool; legal primitive
pushes in unfiltered search are unchanged. In particular, an X permanently frozen
on a winning line rules out that commitment, and reverse-push maps test the
assigned O targets. Multiple-O/OOO handling remains conservative.

The controlled options include serial versus interleaved commitments, straight
versus V3-style forced macros, and a hint-free V3 benchmark adapter. V3 search
code and production models remain unchanged. Removing hints clones the policy
for an offline benchmark and restores the original policy afterward. It does
not change the operational 390-board release baseline.

All lanes share the original 30-second / 500,000-expansion budget. Nested goal
slices obey their parent's remaining time/node quantum. Per-strategy CPU clocks
use the current thread, avoiding attribution of another API thread's work.
Daily selection stays at V3 unless all acceptance gates pass. No saved solution
or official answer is consulted by V4 search.

The developer benchmark endpoint accepts the same commitment/macro switches and
the `beam8` schedule (rank/weighted/macro/beam weights 1:1:1:8). These are
experimental controls, not daily routing changes. Reproduce the strongest
native candidate with:

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --schedule beam8 --output /tmp/v4-beam8.jsonl \
  --gate-v3 debug-artifacts/v4-2026-10-25/v3-baseline.jsonl
```

## Validation protocol

Compare one change at a time with exact input and engine fingerprints. Include
all four prior V3 regressions, the nine production targets, and the complete
397-board corpus. The hint-free V3 run is a diagnostic comparison; production
coverage, no-regression requirements, and the nine-target gate remain intact.
Every positive result must pass the existing independent verifier. Deployment
and repeated fresh/warm-process evidence remain separate from native results.

Numerical receipts and the final outcome are retained under
`debug-artifacts/v4-round2-2026-10-25/`.

## Measured outcomes

All corpus rows use the frozen 397 boards, a 30-second / 500,000-expansion budget,
zero quality-improvement allowance, and independent replay verification. Each
corpus is sequential; independent experiments ran concurrently on the native
host. No CNN fallback or exact action hints are used by V4.

| Configuration | Verified within 30s | Nine targets | V3 coverage losses |
| --- | ---: | ---: | ---: |
| Operational V3 reference | 390/397 verified reference | 4/9 | — |
| V3 with exact hints removed | 375/397 | 3/9 | diagnostic comparison |
| Current V4 rank schedule | 388/397 | 7/9 | 7 |
| Current V4 beam schedule, straight macros | 393/397 | 8/9 | 3 |
| Current V4 beam8, forced macros | 392/397 | 9/9 | 4 |
| Current V4 beam8, straight macros | **395/397** | **9/9** | **2** |

The hint-free V3 run has 376 verified solutions, one finishing after 30 seconds;
375 satisfy the wall budget. This ablation retains feature weights while cloning
away every exact action hint and priority-recovery flag. It explains an important
confounder without lowering the operational 390-board release standard.

Both beam8 macro variants passed three fresh-process and three warm-process
runs per target on the native host: 54/54 each. Straight macros peaked at 26.37s
per solve; forced macros at 24.39s. These are **not** the required ten-plus-ten
deployment-matched repetition gates. Separate local Linux ARM64 probes, capped
at one CPU and 512 MiB, solved 9/9 for each macro mode. Peninsula took 29.43s
with straight macros and 24.61s with forced macros. This runtime margin is small.

The deployed read-only endpoint has the exact implementation engine hash. It
solved Bubble Wrap in 7.58s with straight macros, but Peninsula timed out at
29.75s with both macro modes. The successful native straight run expanded
16,432 states in 26.11s; the deployed straight probe expanded 7,151 in 29.75s.
Deployed thread CPU time was 29.64s, close to wall time. These probes establish
lower useful search throughput on that runtime, not CPU throttling or cold-start
as its cause. HTTP/network time is recorded separately. No daily solve or
archive publication was triggered.

The score-cache microbenchmark on 62 frozen fixtures measures 1.61x, 1.44x,
and 1.22x speedups for requested prefixes of 1, 8, and 24 plans. It uses cold
score caches and warm geometry; it is not a whole-solver speedup. A separate
eight-second profiled Peninsula rank-search probe expanded 4,202 states versus
2,893 before these changes. Scoring remains the largest measured search cost:
4.17s cumulative in the current profiled run, versus 5.27s before. Profiling
changes search work and these timings do not establish solve coverage.

Move quality is measured separately: on 388 boards solved by both the operational
V3 reference and beam8/straight, median keystrokes increase from 53 to 56;
aggregate moves increase from 23,632 to 24,695 (4.5%). Coverage improved, but this
candidate is not uniformly faster or shorter. The existing keystroke optimizer
remains available within the original shared budget.
The best candidate's warm-process native peak RSS was 177.5 MiB; the local Linux
straight-macro probe peaked at 138.3 MiB. These are process high-water marks, not
per-board allocation measurements.

Validation passed locally: 172 Python tests including isolated Postgres
integration, 34 web tests, web lint, and targeted undefined-name/syntax lint.
[Hosted CI](https://github.com/hoverdart/tic-tac-go-perfected/actions/runs/40281714478)
passed Python/Postgres tests, web tests/lint, and the web build at `be10287`.
[Cloud Run deployment](https://github.com/hoverdart/tic-tac-go-perfected/actions/runs/40281714476)
also passed; that workflow result is not a deployed solver benchmark.

The full corpus coverage gate fails on both V3 regressions. The deployed target
gate also fails, and the 180-case deployment-matched repetition matrix has not
been accepted. No schedule, model, daily selector, or archive record is promoted.

## Next search experiment from the research

The immediate blockers are search guidance on Immovable Objects and Cold Feet,
and deployed throughput on Peninsula. Known verified paths and the unchanged
primitive-successor differential tests provide diagnostic evidence; accepting
those saved paths would not satisfy search acceptance. Further optimize the
remaining scoring work with exact ordering fixtures, profile the deployed
engine, and test a schedule change only against the complete corpus. Replacing
the failed schedule with a union of date-specific winners is not an evaluation.

After closing the measured deployment gap, test a feature-policy PHS/Levin lane
against these exact controls. Train push probabilities on independently verified
trajectories; retain a small probability floor for every legal primitive action.
Measure calibration and solved-board coverage, not only action accuracy. Keep
path probability in log space, use a separate frontier/dominance rule, and debit
the existing shared node/deadline budget. Saved action lookup stays disabled.
Finite time, pruning, and duplicate-state handling must be audited independently;
the paper's guarantees are not automatic for a modified graph implementation.

Expand data with verified puzzles generated under Tic-Tac-Go rules, using random
transformations only inside the original identity's split. Boxoban's four-box
fixed-target data is not a compatible replacement. Use all nine hard targets as
development cases, and reserve untouched puzzle identities for validation. A
new model is promoted only after the full corpus, target repetitions, and
deployed search gates pass. Neither this development corpus nor a few recovered
boards establish unseen-board generalization.
