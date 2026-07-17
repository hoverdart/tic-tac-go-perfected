# Tic Tac Go Perfected

Solver for Tic-Tac-Go, a push-puzzle game where an agent maneuvers pieces into a 3-in-a-row configuration with a web frontend and FastAPI backend. Uses BFS/A* for smaller boards and beam search combined with a behavioral cloning CNN trained on 15,000+ board/move pairs for complex 8x8 configurations, selecting the appropriate algorithm based on board size and complexity. Achieves 93% solve rate across 334 real game boards.

## Project Layout

- `apps/web/`: Next.js frontend for showing the board, solve state, and steps.
- `apps/api/`: FastAPI backend scaffold for exposing the solver over HTTP.
- `solver/`: Python solvers, screenshot parsers, model-guided search, and
  training experiments.
- `tictacgo_solver_v3.md`: Push Solver V3 diagnosis, architecture, coverage
  invariant, and measured release results.

## Local Development

### Solver

```bash
cd solver
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 solve.py --quiet-progress
```

### API

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r apps/api/requirements.txt
python3 -m playwright install chromium
uvicorn apps.api.main:app --reload
```

The API exposes:

- `GET /health`
- `POST /solve`
- `POST /jobs/daily-solve`
- `GET /solutions/today`
- `GET /solutions/{date}`

If the frontend is not running on `http://localhost:3000`, set
`API_ALLOWED_ORIGINS` for the backend.

The daily job also needs `DATABASE_URL`, `GEMINI_API_KEY`, `CRON_SECRET`, and
`GOOGLE_TIC_TAC_GO_URL`.

The daily cron job runs Push Solver V3 first. V3 gives the coverage-preserving
V2 portfolio the complete 500,000-node/30-second budget, then uses time left in
the first 10 seconds to improve any verified incumbent by total keystrokes. If
push search does not solve the board, it falls back to the heuristic beam solver
and then CNN-guided beam search. Every returned path is independently verified.

The direct `POST /solve` API remains configurable. Boards that are at least
`6x6` route to the heuristic-CNN beam solver. Smaller boards use the legacy
solver by default, or the optimized pure-Python solver when configured with:

```bash
SOLVER_IMPL=optimized
SOLVER_MODE=hybrid
```

`SOLVER_MODE` can be `hybrid`, `fast`, or `exact`; `hybrid` is the default.
Set `SOLVER_IMPL=learned` to try the Linear Tree Solver V1 on non-large boards.
It loads `solver/learned_search/linear_tree_ranker_v1.json` and ranks candidate
child paths during A* search. Large `6x6+` boards still route to heuristic-CNN by
default.
Set `SOLVER_IMPL=push` to try the classical Sokoban-style push-level solver on
any board size. This is opt-in and bypasses the large-board heuristic-CNN route.
Set `SOLVER_FALLBACK=none` to disable the optimized solver's legacy fallback.

`POST /solve` responses include `solver_name`, such as `bfs`,
`heuristic-CNN`, `push-v3`, or `optimized-hybrid`, so each board records which solver
actually ran.

### Historical unresolved backfill

`backfill_solutions.py` retries only stored rows whose status is not `solved`.
It uses the same push portfolio and 500,000-node/30-second budget as the daily
job, never overwrites an existing solved row, and replaces its failure report
on every run so stale failures do not accumulate.

Load a `DATABASE_URL` in `.env`, then audit the complete historical manifest
with one database query:

```bash
python3 backfill_solutions.py --audit-only
```

The audit reports separate `solved`, `unresolved`, and `missing` counts. Use
`--list-only` to print only the unresolved rows that a normal run would retry.

Import verified paths that already exist in Postgres but are missing from the
training corpus without rerunning any solver:

```bash
python3 backfill_solutions.py \
  --sync-corpus-only \
  --solution-corpus solver/gymnasium_register/all_boards_heuristic_cnn_solutions.jsonl
```

If the local virtual environment is stale, create a temporary Python 3.12
environment with the API's Postgres dependency before running the commands:

```bash
/opt/homebrew/bin/python3.12 -m venv /tmp/tic-tac-go-backfill-venv
source /tmp/tic-tac-go-backfill-venv/bin/activate
python -m pip install -r apps/api/requirements.txt
```

Exercise one board without writing to Postgres:

```bash
python3 backfill_solutions.py \
  --solver push \
  --dry-run \
  --limit 1 \
  --failure-log /tmp/backfill_push_failures.jsonl
```

Run the full backfill and add newly verified push paths to the ranker corpus:

```bash
python3 backfill_solutions.py \
  --solver push \
  --timeout-seconds 30 \
  --max-nodes 500000 \
  --failure-log backfill_push_failures.jsonl \
  --solution-corpus solver/gymnasium_register/all_boards_heuristic_cnn_solutions.jsonl
```

Dates missing from Postgres are skipped unless `--include-missing` is passed.
Use `--start-date`, `--end-date`, or `--limit` to run smaller batches. The
failure JSONL records each unsuccessful strategy and its search diagnostics.

Newly solved paths are valid supervised ranking examples. Remaining failed
boards are useful hard-tail benchmarks, but they do not provide positive
ranking labels by themselves. Train a candidate model outside the tracked
production path first:

```bash
python3 -m solver.push_solver.training_export \
  --solutions solver/gymnasium_register/all_boards_heuristic_cnn_solutions.jsonl \
  --examples-out /tmp/push_ranker_backfill_examples.jsonl \
  --model-out /tmp/linear_push_ranker_candidate.json \
  --verify-solutions
```

Do not replace `solver/push_solver/linear_push_ranker_v1.json` until the
candidate has been benchmarked against the current model.

Set `PUSH_RANK_POLICY_PATH=/path/to/candidate.json` to benchmark a candidate
primary policy without replacing the tracked model. Optional supplemental
policies remain active so the comparison matches production loading behavior.

### Web

```bash
cd apps/web
npm install
npm run dev
```

For Vercel, deploy from the repository root. The root `vercel.json` configures
the Next.js frontend and FastAPI backend as services in one Vercel project.
Set `CRON_SECRET` on the Vercel project.

### Deploy to Vercel

Create one Vercel project from this repository and keep the project root set to
the repository root. Vercel Services will mount:

- `apps/web` at `/`
- `app.py` at `/api/python`

The root `app.py` exports `apps.api.main:app` for Vercel's Python runtime, and
`pyproject.toml` defines the Python dependencies. Vercel's function bundle is
too small for a local Chromium binary, so the daily screenshot job should use a
remote browser endpoint in production.

Set these environment variables on the Vercel project:

- `API_ALLOWED_ORIGINS`: `https://tictacgo.shauryav.com`
- `CRON_SECRET`
- `DATABASE_URL`
- `GEMINI_API_KEY`
- `GOOGLE_TIC_TAC_GO_URL`
- `SOLVER_IMPL`: optional direct `POST /solve` selection; the daily cron uses
  push with Beam/CNN fallback regardless of this setting
- `SOLVER_MODE`: optional, `hybrid`, `fast`, or `exact`
- `REMOTE_BROWSER_PROVIDER`: `browserless`
- `BROWSERLESS_TOKEN`: Browserless API token if using Browserless instead
- `BROWSERLESS_REGION`: optional Browserless region, defaults to `production-sfo`

You can also set `PLAYWRIGHT_CDP_URL` or `BROWSERLESS_WS_URL` directly if you
use another remote browser provider. Remove stale `PLAYWRIGHT_CDP_URL`,
`BROWSERLESS_WS_URL`, `BROWSERBASE_API_KEY`, and `BROWSERBASE_PROJECT_ID` values
from Vercel unless you intentionally use them.

The simplest remote browser option is Browserless BaaS:

1. Create a Browserless project and copy the API token from its dashboard.
2. Set `BROWSERLESS_TOKEN` in Vercel to that token.
3. Redeploy and test `POST /api/manual/daily-solve`.

Browserless REST URLs such as `/pdf` are for one-off HTTP tasks. This app uses
Browserless BaaS over WebSocket/CDP, equivalent to
`wss://production-sfo.browserless.io?token=YOUR_TOKEN`.

You can inspect which remote browser config Vercel selected with:

```bash
curl https://tictacgo.shauryav.com/api/python/debug/remote-browser \
  -H "Authorization: Bearer $CRON_SECRET"
```

You can download the exact screenshot the production capture step sees with:

```bash
curl https://tictacgo.shauryav.com/api/python/debug/screenshot \
  -H "Authorization: Bearer $CRON_SECRET" \
  --output debug-artifacts/google-tic-tac-go.png
```

Do not deploy the browser runner as another Vercel Function. The same bundle
limits apply there too.

The web service uses Vercel's generated `BACKEND_URL` to call FastAPI. You can
still set `API_BASE_URL` to override it manually, but do not set it to
`http://127.0.0.1:8000` or `http://localhost:8000` in Vercel. Those values are
only for local development.

If you set `API_BASE_URL` in production, use
`https://tictacgo.shauryav.com/api/python`.
