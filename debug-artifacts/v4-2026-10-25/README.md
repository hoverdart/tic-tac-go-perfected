# V4 experiments through October 25

V4 remains experimental. Input truth is `catalog.jsonl` (397 boards) and
`source.json`. Reference coverage is `v3-baseline.jsonl` (390 verified), with its
metadata receipt. Historical and parser-repair paths remain in the separate
October 22 audit directory.

Final coverage receipts use `corpus-final-*` after the quality-clock
compatibility correction. `targets-frozen-*`, `corpus-frozen-*`, and
`container-frozen-beam.*` pin the preceding source; quality was disabled in
those coverage probes. They are retained as separate cohorts. Their metadata pins the source and model files. Gate
reports are generated separately and must fail on stale engine hashes, mixed
configurations, research budgets, hints, missing boards, or lost V3 coverage.
The default schedule is `rank` (4:2:1:1); `beam` (1:1:1:4) is an explicit
experimental alternative. The daily version selector still defaults to V3.

`offline-bounded`, `offline-20251228`, `offline-20260314`, `offline-20260913`, and
`offline-20261004` contain independently replayed research paths for all five
persistent hard boards. Their 300-second / 5-million-node limits cannot satisfy
the daily release gate, even if a path happened to appear before 30 seconds.
Separate-board research sweeps ran on separate host cores; they are not isolated
production timing evidence. The older unbounded-cache attempt is retained in
`offline-initial` and was superseded after its memory growth was measured.

`feature-candidate-finite.json` is the corrected, finite, feature-only ranker. Its
manifest keeps transformed board identities together and places all nine target
boards in development. It has no action hints and is not selected in production.
Initial nonfinite candidate fits and their invalid comparisons are quarantined
in `invalid-candidates/`; do not use their ranking metrics.

Other filenames are intermediate experiments. In particular, `v4-corpus-final`
is an earlier, superseded engine despite its original filename. Intermediate
cache/route experiments are not release evidence. Replay-verified paths remain
path evidence only. Their source fingerprints must not be merged into the final
candidate cohort. `bubble-wrap-corrected-local.json` explicitly labels its
unpinned intermediate timing.

`deployed-bubble-wrap.json` and `deployed-targets.jsonl` are authenticated,
read-only probes of the initial pushed engine (a9470d5), served by the actual
Vercel API. No daily row or archive record was changed. Their source hashes are
included; later engine probes are stored separately. The controlled Docker
runs are not claimed to match Vercel CPU quotas, which were not exposed.

`verified-path-corpus.jsonl` indexes independently replayed paths for all 397
boards, with source provenance. Original evidence stays in place. It is for
supervision and verification only, never exact-action search hints.

CPU deltas use `time.process_time`; concurrent work in the same server process
can contribute. Memory fields are process-lifetime RSS peaks, not per-board
allocations. The API probe HTTP time also includes catalog/network overhead.

`bubble-matched-quality.jsonl` uses equal 30-second / 500,000-node budgets and
V3-compatible solve-start quality windows. `bubble_matched_quality.py` reproduces
one fresh-process measurement (`PYTHONPATH=. .venv/bin/python <script> v3|v4`).
Earlier post-incumbent quality-window experiments are superseded. Their shorter
replay-verified paths remain supervision evidence, not a matching-argument
performance comparison.
