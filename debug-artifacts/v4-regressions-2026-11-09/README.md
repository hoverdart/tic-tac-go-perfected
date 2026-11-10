# V4 regression receipts

Implementation and interpretation:
[November 9 regression report](../../docs/audits/push-solver-v4-regressions-2026-11-09.md).

The final native corpus passes **397/397**, **9/9** mandatory targets, and
**zero** V3 coverage losses. Full runtime/repetition release gates remain unmet.

Current-engine receipts:

- `corpus-coverage.jsonl` and its metadata/gate: sequential frozen 397-board
  comparison, 30 seconds / 500,000 expansions, quality disabled, no fallback.
- `focused-coverage.jsonl`: separate three-board fresh-process native run,
  including a failed Immovable Objects attempt; it must not be discarded when
  judging repeatability.
- `container-coverage.jsonl`, `container-environment.json`: one-CPU / 512-MiB
  Linux ARM64 selected-case probe. Both original coverage regressions recover;
  Peninsula Problems remains unsolved under that runtime limit.
- `recent-coverage.jsonl`: the official November 7–8 boards, outside the frozen
  corpus and fitting data; no production writes were performed for these probes.
- `validation.json`: Python, NumPy, test, and lint measurements.
- `summary.json`: aggregate gate, independently replayed paths, runtime and move
  quality comparisons. `before.jsonl` retains the two reproduced old failures.

Other files are superseded diagnostic experiments. In particular,
`corpus-goal-score.jsonl` is an intentionally stopped partial cohort after
replacement-scoring regressions appeared. Files ending in `probe.jsonl` used
in-memory prototype patches: their engine hashes identify the imported base
files, not the complete prototype. They are not release inputs or final-engine
coverage claims. Do not merge different configurations into one result.
The `-16` cohorts use the intermediate beam allocation; that full corpus
recovered both V3 regressions but failed mandatory Peninsula Problems. They are
superseded by the final engine's 1:1:1:32 allocation.

Corpus jobs execute boards sequentially. Independent focused/native and local
container experiments overlap on the host, so these measurements are not
isolated machine-performance comparisons. A full corpus pass does not substitute
for deployed runtime checks or the required repetition matrix. Daily remains V3.

Reproduce the candidate:

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --schedule coverage --output /tmp/v4-coverage.jsonl \
  --gate-v3 debug-artifacts/v4-2026-10-25/v3-baseline.jsonl
```
