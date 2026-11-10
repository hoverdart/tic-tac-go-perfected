# November 7–8 V4 recoveries

Both previously unsolved production boards were recovered with the existing V4
`beam8` portfolio, interleaved committed beam, straight macros, three routes,
30 seconds / 500,000 nodes, and quality optimization disabled. No exact action
hints, fallback solver, policy model, or engine changes were used.

| Official board | Local time | Deployed time | Moves | Pushes |
| --- | ---: | ---: | ---: | ---: |
| Windmill, November 7, 2026 | 1.588s | 4.887s | 77 | 23 |
| Mind The Gap, November 8, 2026 | 0.096s | 2.980s | 91 | 17 |

Google's current official catalog matched both stored layouts. Original run
receipts identify V3 followed by heuristic/CNN fallback, ending unsolved after
roughly 639 and 634 seconds respectively. The stored `heuristic-CNN` solver
label identifies the final fallback, not the whole attempted portfolio.

Authenticated `POST /debug/push-benchmark/{date}` requests with
`{"version":"v4","schedule":"beam8"}` produced the deployed receipts. These
benchmark requests themselves performed no database writes. Each winning path
was then independently replay-verified against the freshly fetched official
board, checked against the reported final board, and replayed into per-move
steps before saving the two records through the existing storage API as
`solver_name=push-v4`. Their original source URLs and official titles were
retained. The existing cache publication endpoint refreshed both dates.

Public solution API reads and both dated HTML pages were checked after
publication; `public-verification.json` records those checks. Original V3 run
ledger entries remain historical failures; this recovery did not rerun or
rewrite those attempts. The deployed benchmark receipts describe the V4 searches.

Evidence:

- `boards.json`: official layouts, source URL, and hashes.
- `stored-*.json`, `db-before-*.json`, `runs-before-*.json`: original failures.
- `daily-budget.jsonl`, `provenance.json`: local runs and source fingerprints.
- `deployed-*.json`: deployed V4 results, paths, diagnostics, and source hashes.
- `publication.json`, `public-verification.json`: publication and live readback.

To reproduce the local searches without writes, run from the repository root:

```python
import json
from pathlib import Path
from solver.push_solver.benchmark_v4 import run

path = Path("debug-artifacts/v4-2026-11-09/boards.json")
for entry in json.loads(path.read_text())["entries"]:
    row = {
        "id": entry["id"], "date": entry["puzzle_date"],
        "name": entry["title"], "width": entry["width"],
        "height": entry["height"], "puzzle": entry["level"],
        "board_sha256": entry["board_sha256"],
    }
    result = run(row, schedule="beam8")
    print(row["date"], result["verified"], result["wall_seconds"])
```

These two recoveries do not pass the full V4 release gates. The latest corpus
control remains 395/397 with V3 coverage regressions; previous deployed hard-case
timeouts also remain unresolved. No daily selector or model was promoted.
