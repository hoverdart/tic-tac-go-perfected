# Recorded outcomes and archive indexing audit

## Measurement scope

The September 8 artifacts are a historical snapshot, not a current solve-rate
claim. They report final stored records; they do not count every unsuccessful
push attempt before a successful fallback. Recorded times exclude screenshot
capture and parsing. Missing timing data is not zero runtime.

`debug-artifacts/production-metrics-2026-09-08.json` contains 351 stored rows,
348 marked solved (99.15%), and three marked unsolved. Of the solved records,
211 have no timing. The 51 successful push-v3 records all report less than 30
seconds, with a 5.08-second median. This is not a 100% push attempt success rate:
54 final records were attributed to push-v3, including three unsolved rows.

**Correctness qualification, October 22:** comparing 394 stored boards with
Google's then-current published layouts found 11 mismatches. Ten records marked
solved have paths that fail against the official layouts. All 386 solved paths
pass against their own stored boards, but only 376 pass against the official
layouts. The older stored-outcome percentage must not be read as an independently
verified official-board success rate. Historical screenshots are unavailable, so
the comparison alone cannot attribute every mismatch to recognition, date
selection, or a later source change.

## Fixed-corpus solver runs

The September full run (`debug-artifacts/push-benchmark-2026-09-08.csv` and its
summary) verified 333 of 341 boards. It improved 208 paths and reduced total
keystrokes on the 333 solved boards from 19,918 to 15,075. The run selected only
push search: empty Beam fields and zero Beam coverage do not measure Beam quality.

The isolated rerun checked five boards. It recovered Thread the Needle and
Yellow Light; Full Steam Ahead, Anchor, and Gears still timed out. Combining
these isolated rows with the full run gives 335/341. The earlier checked-in July
V3 release CSV has 337/341. Different wall-clock outcomes require isolated
reruns before concluding that algorithmic coverage changed. Do not overwrite
either baseline or silently combine their timings.

## Indexing evidence

The saved before/after HTTP audits each cover 357 URLs. Before, all returned 200;
two carried noindex. Historical pages had no archive links in their initial
HTML. After, two missing URLs returned 404 and most dated pages contained 15
archive links before JavaScript. This proves HTTP/HTML behavior at capture time;
it does not prove Google indexed the pages or improved their ranking.

The archive changes provide hourly revalidation for dated data and history,
preserve errors during failed regeneration, omit pending/invalid sitemap dates,
and remove invented daily modification dates from the evergreen guides. Optional
history may fail without hiding an otherwise available solution. Client refresh
requests the complete archive rather than replacing old neighbors with a
365-record window. The loading boundary is removed so missing routes can return
404 before streaming response headers.

## Reproduction

Run `npm test`, `npm run lint`, and `npm run build` in `apps/web`. For recorded
database metrics, `production-solver-metrics.sql` uses a read-only transaction.
The full and isolated benchmark summaries retain their original denominators.
See `tictacgo_solver_v3.md` for benchmark and coverage-guard commands.
