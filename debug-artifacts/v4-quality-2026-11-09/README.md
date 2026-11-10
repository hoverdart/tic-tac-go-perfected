# V4 move-quality optimization receipts

Interpretation and research:
[November 9 quality report](../../docs/audits/push-solver-v4-quality-2026-11-09.md).

Both final native cohorts pass **397/397**, **9/9** mandatory targets, and the
original V3 no-coverage-loss gate. Mean moves are **40.63** with the 10-second
quality window and **39.17** with the 30-second window, versus the previous
coverage control's **66.20**. Total time remains 30 seconds and total expansions
500,000. No exact action hints, archived answers, or fallback solvers enter V4.

Receipts:

- `corpus-quality-10.jsonl`, `corpus-quality-30.jsonl`, metadata and gates:
  complete, separately fingerprinted final-engine cohorts. Do not merge them.
- `v3-quality-corpus.jsonl`: fresh operational V3 with matching 10-second quality
  settings, 383/397 verified. The gate remains the original 390-board reference.
- `corpus-basic-quality.jsonl`: complete preceding rewrite/weighted-search
  control, 397/397 at 44.05 mean moves, before adding the neighborhood phase.
- `focused-basic-quality.jsonl`: selected intermediate live tests.
- `path-rewrite-probe.jsonl`, `path-rewrite-half-second-probe.jsonl`,
  `vicinity-probe.jsonl`: offline post-processing experiments on saved coverage
  outputs. These measure optimizers only, not fresh solver coverage.
- `v3-quality-sample.jsonl`: preliminary every-fortieth-board V3 sample.
- `recent-quality-30.jsonl`: read-only tests on the official November 7–8 boards.
- `linux-quality-30.jsonl`, `linux-environment.json`: selected one-CPU / 512-MiB
  local Linux ARM64 probe. Both original regressions recover with shorter paths;
  Peninsula Problems still times out. This is not deployment-matched acceptance.
- `validation.json`: runtime versions and 199-test validation with four optional
  database skips.
- `controls/`: exact prior runtime sources and diagnostic prototype scripts.
- `summary.json`: independent full-path/final-board/push-trace verification,
  expansion accounting, complete aggregates, and paired V3 comparisons.

Each corpus runs its boards sequentially. Independent experiments overlap on the
host; they are development measurements, not isolated speed comparisons or an
unseen-board evaluation. The full deployment/repetition gates remain open and
the daily selector stays V3. No production archive records were updated.

Reproduce the final candidate:

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --schedule coverage --quality-seconds 30 \
  --output /tmp/v4-quality-30.jsonl \
  --gate-v3 debug-artifacts/v4-2026-10-25/v3-baseline.jsonl
```

Use `--quality-seconds 10` for the matching quality allocation. Rebuild all
aggregates without running the solver again:

```sh
PYTHONPATH=. .venv/bin/python debug-artifacts/v4-quality-2026-11-09/summarize.py
```
