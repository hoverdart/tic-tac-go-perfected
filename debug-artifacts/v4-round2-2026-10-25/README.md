# V4 round-two receipts

Implementation: `be10287`. Research and interpretation:
[Google research and controlled search changes](../../docs/audits/push-solver-v4-google-research-2026-10-25.md).

The current best native candidate is `corpus-beam8-straight.jsonl`: 395/397 and
9/9 targets. The failed coverage gate records the two remaining V3 regressions.
This configuration uses no exact action hints. Daily remains V3.

Primary comparable native cohorts:

- `corpus-round2-default.jsonl`: rank schedule, straight macros, 388/397.
- `corpus-round2-straight.jsonl`: beam schedule, straight macros, 393/397.
- `corpus-beam8-forced.jsonl`: beam8 schedule, forced macros, 392/397, 9/9 targets.
- `corpus-beam8-straight.jsonl`: beam8 schedule, straight macros, 395/397, 9/9.
- `v3-hint-free.jsonl`: original V3 feature weights with every exact hint and
  priority-recovery flag removed offline; 376 verified, 375 within 30 seconds.
  This does not replace the operational 390/397 baseline.

The two beam8 configurations have separate three-fresh/three-warm native target
cohorts: 54/54 each. Do not merge different macro modes, schedules, model hashes,
or engine fingerprints into a release result. The required 180-case
container/deployment-matched repetition gate is not satisfied by these files.

`container*-targets.jsonl` are local Linux ARM64 one-CPU/512-MiB probes. They use
the repository Dockerfile; image identity is in `container-image.json`. They are
not deployment-matched evidence. `deployed*.json` are authenticated read-only
Vercel benchmark responses: Bubble Wrap passed; Peninsula failed for both macro
modes. Authentication values are not stored. Hosted CI and Cloud Run deployment
workflow receipts are separate and do not establish deployed solver coverage.

`scoring-microbenchmark.json` and `peninsula-profile.json` are diagnostic timing
experiments. Neither establishes coverage or unseen-board performance. The old
engine is available at `f02cc37:solver/push_solver/v4.py`.

Other `focused-*`, `targets-*v1/v2`, `targets-budgeted-*`, `targets-final-*`,
`corpus-final-forced`, and feature-ablation files retain intermediate or negative
experiments. Their source fingerprints may differ from the final implementation;
use their own metadata. No intermediate cohort is a release gate for the current
engine. No future catalog boards were used. No archive records were republished.

Each corpus runs sequentially; independent experiments ran concurrently on the
native host. Limits are 30 seconds / 500,000 expansions without fallback or a
quality-improvement allowance. Data is development evidence, not an unseen-board
generalization result.

Regenerate the aggregate without mixing configurations:

```sh
PYTHONPATH=. .venv/bin/python debug-artifacts/v4-round2-2026-10-25/summarize.py
```

Reproduce the candidate and its failed full-corpus gate:

```sh
.venv/bin/python -m solver.push_solver.benchmark_v4 \
  --schedule beam8 --output /tmp/v4-beam8-straight.jsonl \
  --gate-v3 debug-artifacts/v4-2026-10-25/v3-baseline.jsonl
```
