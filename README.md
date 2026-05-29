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

For Vercel, set the project root directory to `apps/web`.

The web app needs `API_BASE_URL` so it can read daily solutions from FastAPI.
Set the same `CRON_SECRET` in both Vercel and the FastAPI host.
