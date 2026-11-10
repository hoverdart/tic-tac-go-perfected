# V4 production qualification — November 9, 2026

The initial release candidate solves all **397/397** frozen official boards in a
one-CPU, 512 MiB Linux container. Mean total direction keystrokes are **38.36**,
median 31, and the longest solve is **17.87 seconds**. Independent replay checks
every path and final board. The original V3 reference remains 390/397; the
coverage gate reports no lost V3 solutions and all nine mandatory targets pass.

The earlier [quality report](push-solver-v4-quality-2026-11-09.md) records the
research and separate native cohorts, including the matched V3 comparison.
Those historical measurements have not been overwritten by this release run.

## Kernel and configuration

The daily V4 configuration is `schedule="coverage"`, 30 seconds total,
500,000 expansions, and a quality window ending at solve-start plus 30 seconds.
The optimizer uses only the current invocation's verified incumbent. It never
loads archived answers or exact hints.

Production profiling identified repeated route selection, fixed-point frozen
piece tests, and O mobility counts. The release precompiles static opposing-axis
masks and caches route choices by only the relevant X intersections. It preserves
the reference tie order. Randomized fixed-point fixtures and frozen scoring
fixtures check semantic equivalence. Three constrained Linux hard probes take
15.41 seconds (Immovable Objects), 8.79 (Cold Feet), and 16.92 (Peninsula Problems).

Kernel SHA-256:
`bab886c4b478abf071a7e5c97aa75d5e0cdeb7afb1a30a29202297625410e4d1`.

Runner SHA-256:
`9e5604875e41fa580f5e8754d533981d95083b4b346dc6d4e369698c4cb4e3f2`.

The full Linux corpus started before the fresh-worker transport additions;
its metadata retains that earlier runner fingerprint. Its five-file kernel is
identical to this release. The separate repetition matrices use the final
runner and independently fingerprint both kernel and runner. All **180/180**
local receipts pass, with a maximum fresh-process time of **17.41 seconds**
including startup and a maximum warm solve of **17.94 seconds**.

## Deployment and validation

The website calls **Cloud Run**, not the Vercel Python function, for daily solves.
Local constrained-container results therefore cannot alone establish production
readiness. The Vercel preview builds and returns the correct fingerprints, but
its Immovable Objects probe times out at 29.92 seconds. This failed receipt is
retained in `preview-probes.jsonl`; it is not counted as a pass or silently retried.

The initial code deployment retains V3 as the daily default. Actual Cloud Run
qualification must precede promotion: one complete 397-board corpus, the nine
mandatory targets repeated ten times in each of fresh and warm processes
(180 receipts), and both original regressions repeated three times per process
mode. Every cohort uses the same configuration and records its actual revision.

`GET /debug/solver-config` reports authenticated, uncached daily settings and
runtime/source identity. The read-only benchmark accepts `process="fresh"`;
this launches a fixed Python worker with JSON input, includes startup in its
30-second process limit, and reads no saved corpus in the child. Failures remain
in the output cohort. Neither diagnostic endpoint writes solution records.

The deployment workflow now waits for successful validation of the exact main
commit and skips superseded candidates before authenticating to Google Cloud.
`actionlint` passes. Packaging excludes local environment files, Vercel metadata,
and Google authentication credential files. The Vercel dry package contains
156 files / 10.31 MiB with the required runtime model and policy assets.

Both original regressions also pass all twelve additional local trials: three
fresh and three warm solves each, with the final Docker image's dependencies.
Fresh-worker wall time includes Python startup. These extra guards are kept
separate from the mandatory nine-board repetition matrix.

Local validation passes **205 Python tests against PostgreSQL with no skips**,
**34 web tests**, web lint, Docker image build, and the production Next.js build
against an isolated local API and PostgreSQL database. No production database
was used for tests. Kernel lint checks for syntax/undefined-name errors and
`git diff --check` pass. Full style lint has existing import/order diagnostics
and intentionally does not rewrite a fingerprinted benchmark candidate.

## Compiled follow-up candidate

Second-generation execution alone still times out on Peninsula Problems and
Immovable Objects. Its complete failed cohort is retained in `cloudrun-508f670`.
The deployment confirms **one CPU, 4 GiB RAM, concurrency 1**. Local 512 MiB
measurements were conservative controls, not an exact resource match.

The follow-up kernel reuses completed macro successor lists, bounds coverage
caches at 65,536 entries, batches small cooperative checkpoints, and allocates
the three coverage beams by 32 expansions rather than short wall-time slices.
Every node boundary still yields and checks the shared deadline/node cap;
strict one-expansion scheduler slices remain tested. Cache publication is atomic
at completion, and callers receive separate lists. Optional feature rankers
retain their uncached path. The larger caches apply only to the coverage portfolio.

Docker compiles the same two Python kernels with pinned Cython 3.3.1 and
`annotation_typing=False`, using a separate build stage. The runtime image
contains the extensions and source files, without the compiler toolchain.
[Cython's pure Python documentation](https://docs.cython.org/en/latest/src/tutorial/pure.html)
describes this supported source workflow and disabling annotation-based typing.
No fixed-width replacement for arbitrary-precision puzzle masks is introduced.

Runtime provenance includes compiler version, flags, builder/source hashes,
architecture, and both binary hashes. The runtime validates source and binary
hashes before reporting a compiled build. Release gates reject mixed native
artifacts; the hosted client independently checks the actual fresh worker's
implementation against the deployed manifest.

Both Python and compiled images pass **209 tests against PostgreSQL with no
skips**. All eleven local compiled smoke boards pass, with a longest solve of
11.08 seconds in the initial constrained run. The new complete local corpus,
repetitions and hosted checks must finish before promotion.

Follow-up kernel SHA-256:
`d5ef67ca4fd81415b224b6c0f79b15a804b79e7a00f855cb1095af4d33513c84`.

Follow-up runner SHA-256:
`f0de0ef90fddf36324a65dc15ec87d9c85a35bedf16557649e3fab0f3bda0191`.

## Reproduce hosted qualification

### Initial hosted failure and execution environment follow-up

The first code deployment (`4175abb`, Cloud Run revision `00055-2tg`) passes CI
but fails the hosted smoke gate: March 14, Peninsula Problems, and Immovable
Objects time out. The full matrix and corpus do not run after that failed gate.
V3 stays selected. The complete failed cohort is retained in `cloudrun-4175abb`.

Local checkpoint batching, quadrupling the score cache, and compiling the Python
modules provide only modest improvements. Vectorizing refined route selection
also fails to produce a material speedup. These explicitly monkeypatched
experiments are diagnostic controls, not release receipts. Their scripts and
outputs are under `experiments/`; no experimental kernel was promoted.

Cloud Run's [execution environment documentation](https://docs.cloud.google.com/run/docs/configuring/execution-environments)
recommends its second-generation microVM for CPU-intensive workloads and states
that it provides faster sustained CPU performance and full Linux compatibility.
The follow-up deployment explicitly selects `gen2`, retains the same kernel,
budgets and V3 daily selection, and records CPU/memory/concurrency settings.
Its complete cohorts must pass on that new revision before promotion. Local
development hardware is not treated as a substitute for that evidence.

### Beam-width follow-up

The three-lane compiled candidate passes its complete local 397-board corpus
(38.26 mean moves), all 180 repetitions and twelve regression guards. Its hosted
smoke still fails Immovable Objects, so no promotion follows. All failures and
timings are retained under `cloudrun-24fa433`.

Route-count and beam-width ablations identify complementary narrow frontiers.
The next coverage portfolio shares expansions equally among global widths
16, 32 and 24, the focused width-48 goal stream, and the existing width-96
stream. Each narrow stream retains wider restart phases. The daily V4 settings
explicitly choose two static routes. These choices apply to every board; there
is no date, board-hash, or saved-path condition.

The final compiled local smoke passes all eleven boards, with the slowest at
5.58 seconds. Both Python and compiled PostgreSQL suites still pass 209 tests,
including exact route configuration and all five lanes sharing one node budget.
An increased finish reserve leaves room within the same 30-second limit for
releasing the larger caches and recording the result.

Kernel: `81977675498a97dbf236d47e51d2a68c603e31971a7fc8398d64d1e925a9050e`.
Runner: `9bf23c9c58e1f62f4dc083b08b3f1a2063990a710759d95780f675fe18e13fb3`.
The complete current cohorts are recorded separately in `release-linux`.

Deployment now compares runtime inputs with the deployed source commit before
rebuilding. Evidence/documentation-only pushes preserve the tested image rather
than replacing it during qualification. Runtime configuration/workflow changes
still trigger deployment. The actual hosted gates remain required.

With the existing cron secret available locally:

```sh
PYTHONPATH=. .venv/bin/python scripts/verify_v4_release.py \
  --api-base https://tic-tac-go-834303913000.us-west2.run.app \
  --output-dir /tmp/v4-hosted-release --mode smoke --require-native --routes 2
```

Use `--mode matrix`, `--mode guards`, and `--mode corpus` for complete cohorts.
The client checks deployed source hashes, revision, board identity and budgets,
then independently replays each path. Resuming a different revision requires a
new output directory. A failed receipt is retained rather than retried into a
passing cohort.

Release receipts and runtime metadata are in
[`debug-artifacts/v4-production-2026-11-09/`](../../debug-artifacts/v4-production-2026-11-09/).
