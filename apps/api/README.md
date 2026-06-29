# Tic Tac Go API

FastAPI scaffold for exposing the solver to the Next.js frontend.

## Module Layout

- `main.py`: FastAPI routes, request models, auth dependency, and response shaping.
- `daily_solve.py`: daily capture → parse → solve → persist orchestration.
- `board_capture.py`: browser/remote-browser screenshot capture.
- `board_parser.py`: API adapter around the Gemini board parser.
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
uvicorn apps.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API docs.

Set `API_ALLOWED_ORIGINS` as a comma-separated list when the frontend is hosted
somewhere other than `http://localhost:3000`.

Set these variables before running the daily job:

- `DATABASE_URL`
- `GEMINI_API_KEY`
- `CRON_SECRET`
- `GOOGLE_TIC_TAC_GO_URL`

The API chooses a solver per board. Boards that are at least `6x6` route to the
heuristic-CNN beam solver. Smaller boards default to the legacy solver. To use
the optimized solver for smaller boards, set `SOLVER_IMPL=optimized`;
`SOLVER_MODE` can be `hybrid`, `fast`, or `exact` and defaults to `hybrid`.
The optimized path falls back to the legacy solver unless
`SOLVER_FALLBACK=none` is set. `POST /solve` includes `solver_name` in the
response so callers can see which solver actually ran.

## Endpoints

- `GET /health`: basic service health check.
- `POST /solve`: accepts a board and returns the solver name, move string,
  final board, replay steps, state count, and elapsed time.
- `POST /jobs/daily-solve`: protected cron endpoint that captures, parses, solves, and stores today's board.
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
