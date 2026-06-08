# Tic Tac Go Perfected

Fast Tic Tac Go solver with a web frontend and a planned FastAPI backend.

## Project Layout

- `apps/web/`: Next.js frontend for showing the board, solve state, and steps.
- `apps/api/`: FastAPI backend scaffold for exposing the solver over HTTP.
- `solver/`: Python solver, screenshot parsers, and training experiments.

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
