# Tic Tac Go API

FastAPI scaffold for exposing the solver to the Next.js frontend.

## Run Locally

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r apps/api/requirements.txt
uvicorn apps.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API docs.

Set `API_ALLOWED_ORIGINS` as a comma-separated list when the frontend is hosted
somewhere other than `http://localhost:3000`.

## Endpoints

- `GET /health`: basic service health check.
- `POST /solve`: accepts a board and returns the move string, final board, replay steps, state count, and elapsed time.
