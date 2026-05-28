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
