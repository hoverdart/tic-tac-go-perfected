# Catalog ingestion and V4 evidence, October 22, 2026

## Input contract

Daily acquisition now selects the requested date from Google's published Tic Tac
Go catalog. It validates the date key against the embedded date, dimensions,
cell count, alphabet, player count, and title. Unsupported, partial, and duplicate
entries fail explicitly. Multiple O pieces are allowed: three historical boards
have more than two. The JS is parsed as data and never executed.

Published board coordinates remain unchanged in storage. `parser_name=catalog`
and the versioned bundle URL identify the source; logs include source and board
SHA-256 hashes. The frontend trims only fully blocked outer borders, and hints
use the same visible coordinates. Internal barriers remain intact.

A successful process-local catalog expires after one hour; network failures have
30 seconds of retry backoff and are not cached as successful empty catalogs.
Titles share this loader. The bundle format and module name are undocumented;
format changes may require an extractor update.

If catalog acquisition fails, the existing browser capture and Gemini parser
remain available. The capture must expose one unambiguous visible date matching
the requested date before Gemini is called. Historical and midnight mismatches
are rejected. `GEMINI_MODEL` controls the fallback model; structural validation
still cannot independently establish visual accuracy. OpenCV experiments are
archived and removed from CLI/runtime selection and requirements.

Daily persistence uses a conditional upsert to prevent a late failed/unsolved
retry from overwriting a solved row. Explicit repair/backfill calls retain their
replacement behavior. This is status protection, not a claim that every existing
solved record is correct. No historical production records were changed here.

## Evidence files

`debug-artifacts/catalog-audit-2026-10-22/` contains:

- `catalog.jsonl`: 394 official layouts through October 22, with dates and hashes.
- `stored-records.jsonl`: public production inputs, moves, statuses and measurements;
  bulky replay frames are omitted.
- `archive-audit.json`, `mismatches.json`, `targets.json`: per-date comparisons,
  independent path checks, and the 18 unresolved/invalid targets.
- `verified-solutions.jsonl`: 376 original stored paths verified on official inputs.
- `solver-runs.jsonl`: all 18 local sequential push-v3 coverage attempts, including
  returned paths, per-strategy nodes, timeouts, and measurements.
- `recovered-solutions.jsonl`: 13 newly verified local paths, kept separate from
  production receipts and suitable for later supervised training.
- `baseline-summary.json`: 13 recoveries and five remaining timeouts.
- `prior-local-experiments.json`: the earlier full-budget weighted-search ablation
  and initial local runs, including their instrumentation qualification.
- `profile-oct18.json`: attributed function costs; cProfile changes deadline
  outcomes, so these are bottleneck evidence, not ordinary latency measurements.
- `metadata.json`: source URL/hash, capture time, environment, budget, and limits.

Future catalog entries are excluded from training and benchmark exports. The
original 376 paths plus 13 local recoveries give known verified paths for 389 of
394 boards. This is mixed historical/local path availability, not a measured
389/394 success rate for the current solver under one budget.

## Remaining search targets

| Date | Official title | Sequential V3 coverage run |
|---|---|---|
| 2025-12-12 | -_- | timeout, 2,952 expanded nodes |
| 2025-12-28 | Surrounded | timeout, 17,448 expanded nodes |
| 2026-03-14 | Tee Off | timeout, 14,352 expanded nodes |
| 2026-09-13 | Peninsula Problems | timeout, 10,987 expanded nodes |
| 2026-10-04 | Cup of Tea | timeout, 14,726 expanded nodes |

Each received 500,000 nodes and 30 seconds, with quality optimization disabled.
The weighted-search-only ablation also timed out on all five with the whole
30-second budget. Timeouts do not establish impossibility. Detailed V4 design
is deferred; the search implementation and ranker models are unchanged.

All ten falsely solved production boards now have verified local replacement
paths. Bottleneck, Here and There, and Walled Off also recover locally. These
results do not establish the cause of the historical deployment failures.

## Validation and reproduction

The local checks passed 136 Python tests, including three real Postgres integration
checks in a disposable schema, plus 34 web tests, ESLint, and a Next.js production
build. The October 21 catalog → solver → local Postgres pipeline returned a
verified 15-move solution with normal V3 quality optimization enabled. Browser QA
rendered 36 cells with the one internal wall and replayed all 15 moves to SOLVED
without JavaScript errors. The live capture fallback also read October 22's
visible date correctly; no paid Gemini call was needed for these checks.

Run the read-only audit and optional local target benchmark from the repository
root:

```bash
.venv/bin/python -m scripts.audit_catalog \
  --through 2026-10-22 --output-dir /tmp/tictacgo-audit --solve-targets
```

Use `--records PATH` and `--catalog-source PATH` to consume saved inputs. The tool
never writes to Postgres or invokes Gemini. Use a fresh output directory to retain
old measurements. Tests can run against an isolated test database with
`TEST_DATABASE_URL=... .venv/bin/python -m unittest discover -s tests`.

The authenticated `GET /debug/catalog/YYYY-MM-DD` endpoint returns the resolved
board and hashes without solving, capturing, or persisting anything. It permits
read-only deployment verification without rerunning a production daily job.
