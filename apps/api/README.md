# Tic Tac Go API

FastAPI scaffold for exposing the solver to the Next.js frontend.

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

## Endpoints

- `GET /health`: basic service health check.
- `POST /solve`: accepts a board and returns the move string, final board, replay steps, state count, and elapsed time.
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
`BROWSERBASE_API_KEY`, and optionally `BROWSERBASE_PROJECT_ID`.

The Vercel Python function bundle cannot fit a bundled Chromium binary. In
production, set Browserbase credentials or one of the remote browser endpoint
variables so the daily job connects to an external Chromium instance instead of
launching a local browser.
For Browserbase, the API creates a session and connects to its returned
`connectUrl`.
For Browserless BaaS, set `PLAYWRIGHT_CDP_URL` to the dashboard-provided
WebSocket URL, such as `wss://production-sfo.browserless.io?token=YOUR_TOKEN`.

After deployment, the API is available under `/api/python`, for example
`/api/python/health` and `/api/python/solutions/today`.
