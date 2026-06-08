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
`pyproject.toml` defines the Python dependencies plus the Playwright Chromium
install step.

Set these environment variables on the Vercel project:

- `API_ALLOWED_ORIGINS`: `https://tictacgo.shauryav.com`
- `CRON_SECRET`
- `DATABASE_URL`
- `GEMINI_API_KEY`
- `GOOGLE_TIC_TAC_GO_URL`

The web service uses Vercel's generated `BACKEND_URL` to call FastAPI. You can
still set `API_BASE_URL` to override it manually, but do not set it to
`http://127.0.0.1:8000` or `http://localhost:8000` in Vercel. Those values are
only for local development.

If you set `API_BASE_URL` in production, use
`https://tictacgo.shauryav.com/api/python`.
