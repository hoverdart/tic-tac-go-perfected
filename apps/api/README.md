# Tic Tac Go API

FastAPI scaffold for exposing the solver to the Next.js frontend.

## Module Layout

- `main.py`: FastAPI routes, request models, auth dependency, and response shaping.
- `daily_solve.py`: daily catalog → solve → persist orchestration, with date-checked screenshot fallback.
- `board_capture.py`: browser/remote-browser screenshot capture.
- `board_parser.py`: catalog-first board resolution and Gemini fallback.
- `puzzle_catalog.py`: validated date-keyed official boards, versioned source provenance, and expiring cache.
- `solution_storage.py`: Postgres persistence for daily solution records.
- `puzzle_titles.py`: live and historical puzzle title lookup.

The old module names (`acquisition.py`, `parser.py`, `storage.py`,
`title_fetcher.py`, and `daily_job.py`) are compatibility wrappers.

## Run Locally

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r apps/api/requirements.txt
python3 -m playwright install chromium
python3 -m apps.api.migrate
uvicorn apps.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API docs.

Set `API_ALLOWED_ORIGINS` as a comma-separated list when the frontend is hosted
somewhere other than `http://localhost:3000`.

Set these variables before running the daily job:

- `DATABASE_URL`
- `GEMINI_API_KEY`: required only when the catalog is unavailable and Gemini fallback runs
- `GEMINI_MODEL`: optional model override for that fallback
- `CRON_SECRET`
- `GOOGLE_TIC_TAC_GO_URL`

The Docker image and `dev.sh` apply the database migration before starting the
API. For other deployment methods, run `python3 -m apps.api.migrate` once per
release. Normal API requests never run schema changes.

Database connections use a lazy process-local pool. Optional settings are
`DB_POOL_MAX_SIZE` (default `5`), `DB_POOL_TIMEOUT_SECONDS` (default `10`), and
`DB_POOL_MAX_IDLE_SECONDS` (default `60`). Public solution reads are cached in
each API process for `SOLUTION_CACHE_TTL_SECONDS` (default `300` seconds), and
writes invalidate that process's cache.

The daily job obtains the requested date from Google's published catalog without
starting a browser or calling Gemini. It records `parser_name=catalog` and the
versioned bundle URL. Successful catalogs refresh after an hour; transient failures
retry after a 30-second backoff. Missing or invalid catalog data falls back to
Gemini only if the captured page has one visible date matching the requested date.
Wrong or unknown screenshot dates fail explicitly. OpenCV is archived. A failed
retry cannot overwrite a solved record; explicit verified repairs remain possible.

`GET /debug/catalog/YYYY-MM-DD` (cron bearer token required) returns the exact
catalog board, board/source hashes, and versioned bundle URL. It performs no
capture, solver run, or database write. Install `tests/requirements.txt` for the
HTTP endpoint tests; set `TEST_DATABASE_URL` for the isolated Postgres tests.

The daily job uses the classical push portfolio first, with a 500,000-node and
30-second budget, then runs heuristic beam search with CNN guidance as a
fallback. The fallback move string is independently verified before storage.

The direct `POST /solve` endpoint chooses a solver per board. Boards that are at
least `6x6` route to the heuristic-CNN beam solver unless `SOLVER_IMPL=push` is
explicitly set. Smaller boards default to the legacy solver. To use the
optimized solver for smaller boards, set `SOLVER_IMPL=optimized`; `SOLVER_MODE`
can be `hybrid`, `fast`, or `exact` and defaults to `hybrid`.
The optimized path falls back to the legacy solver unless
`SOLVER_FALLBACK=none` is set. `POST /solve` and stored daily records include
`solver_name` so callers can see which solver path ran.

## Endpoints

- `GET /health`: basic service health check.
- `POST /solve`: accepts a board and returns the solver name, move string,
  final board, replay steps, state count, and elapsed time.
- `POST /jobs/daily-solve`: protected cron endpoint that resolves, solves, and stores today's board.
- `GET /solutions/today`: today's stored solution or a pending response.
- `GET /solutions/{date}`: stored solution for `YYYY-MM-DD`.

## Deploy to Vercel

Deploy from the repository root as one Vercel project. The root `vercel.json`
configures this FastAPI backend as a Vercel Service mounted at `/api/python`,
next to the Next.js frontend mounted at `/`.

The root `app.py` file exports this FastAPI app for Vercel, and the root
`pyproject.toml` provides the Python dependencies. Configure the Vercel project
with `API_ALLOWED_ORIGINS=https://tictacgo.shauryav.com`, `CRON_SECRET`,
`DATABASE_URL`, `GEMINI_API_KEY`, `GOOGLE_TIC_TAC_GO_URL`,
`REMOTE_BROWSER_PROVIDER=browserless`, and `BROWSERLESS_TOKEN`. Optionally set
`SOLVER_IMPL=optimized` and `SOLVER_MODE=hybrid` to use the optimized solver on
smaller boards.

The Vercel Python function bundle cannot fit a bundled Chromium binary. In
production, set Browserless credentials or another remote browser endpoint so
the daily job connects to an external Chromium instance instead of launching a
local browser.
For Browserless BaaS, set `BROWSERLESS_TOKEN`; the API builds the WebSocket URL
`wss://production-sfo.browserless.io?token=YOUR_TOKEN`.
Use `/debug/remote-browser` with the cron bearer token to confirm which provider
is selected without exposing secrets.

After deployment, the API is available under `/api/python`, for example
`/api/python/health` and `/api/python/solutions/today`.
