# V4 PHS experiment receipts

Implementation: `04d5cf5`. Interpretation and decision:
[Policy-guided search experiment](../../docs/audits/push-solver-v4-phs-experiment-2026-10-26.md).

This is an opt-in experiment. Daily remains at its V3 default. No policy, schedule,
archive record or exact-action lookup was promoted.

The final-engine comparisons are `corpus-control-frozen.jsonl` (395/397, 9/9)
and `corpus-adaptive.jsonl` (393/397, 9/9). Both retain failed coverage gates.
The adaptive policy recovers Cold Feet but loses November 16, February 19,
March 1 and August 23 against V3. Do not combine winners from different runs into
a single-budget coverage result.

`summary.json` records all cohorts and flags their engine fingerprints.
`control-vs-adaptive.csv` compares every board's hash, verification, timing,
expansions, moves and selected schedule. Earlier files, including some with
`final` in their historical names, are intermediate source snapshots; their
stored gates are not acceptance receipts for the current engine.

`decisions.npz` contains 5,176 demonstrated decisions, 46,188 action-feature
rows, group offsets, labels and source-board indices. It has no executable model
or saved-action lookup. The original and reproduced manifests retain official
board identities, splits, verified path hashes, source hashes and code provenance.
All four arrays and the entire compressed dataset reproduce byte-for-byte under
the final feature extractor (`dataset-equivalence.json`).

`policy.json` is the 33-weight linear candidate. `policy-mlp.json` is the 32-unit
tanh candidate. Only the model schema and numeric parameters enter runtime;
training reports and identity manifests remain separate. Unknown fields,
state-action hints, bad shapes and nonfinite coefficients are rejected.
`policy-uniform.json` and `policy-mlp-temperature-one.json` are explicit controls.

The direct neural PHS* probes recover Cold Feet and Peninsula. Their fresh/warm
repetitions pass 12/12; the local one-CPU/512-MiB ARM64 container also solves both.
These selected-case probes are not the required 180-case, deployment-matched
acceptance matrix. The exact-engine/model Vercel probes time out on both.
Hosted CI/deployment receipts are separate from solver runtime coverage.

`eager-prototype*` records an isolated runtime-monkeypatch experiment that accepts
a generated goal immediately. It saves only one expansion in each recovery and
was not added to the runtime. `immovable-offline*` uses a 300-second / five-million
node research budget. Beam finds a longer verified path after 204 seconds;
neural PHS* remains unsolved. Neither is a daily-budget success.

`current-day-probe.json` is October 26's official Left Or Right, inspected after
both models were fixed. All tested modes solve it. It is outside fitting and
calibration and does not constitute generalization evidence.

Regenerate aggregates:

```sh
PYTHONPATH=. .venv/bin/python debug-artifacts/v4-phs-2026-10-26/summarize.py
```

Reproduce the learned recovery experiment:

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --ids 20260802 20260913 --strategy phs_star \
  --policy debug-artifacts/v4-phs-2026-10-26/policy-mlp.json \
  --output /tmp/phs-recoveries.jsonl
```

Reproduce the final adaptive corpus comparison (the release gate should fail):

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --schedule phs_adaptive \
  --policy debug-artifacts/v4-phs-2026-10-26/policy-mlp.json \
  --output /tmp/phs-adaptive.jsonl \
  --gate-v3 debug-artifacts/v4-2026-10-25/v3-baseline.jsonl
```

Refit the neural policy from the committed dataset:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m solver.push_solver.train_push_policy \
  --dataset debug-artifacts/v4-phs-2026-10-26/decisions.npz \
  --architecture mlp --epochs 120 --output /tmp/phs-policy.json
```

Re-export the dataset, without overwriting the recorded artifact:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m solver.push_solver.train_push_policy \
  --dataset /tmp/phs-decisions.npz --workers 4 --paths \
  debug-artifacts/v4-2026-10-25/verified-path-corpus.jsonl \
  debug-artifacts/v4-2026-10-25/v3-baseline.jsonl \
  debug-artifacts/v4-round2-2026-10-25/corpus-beam8-straight.jsonl
```
