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
uvicorn apps.api.main:app --reload
```

The API currently exposes `GET /health` and `POST /solve`.

If the frontend is not running on `http://localhost:3000`, set
`API_ALLOWED_ORIGINS` for the backend.

### Web

```bash
cd apps/web
npm install
npm run dev
```

For Vercel, set the project root directory to `apps/web`.
