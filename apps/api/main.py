"""
FastAPI application — public HTTP API for the Tic Tac Go solver.

Routes:
  GET  /health                  — liveness probe
  GET  /debug/remote-browser    — dump remote-browser diagnostic info (cron-auth required)
  GET  /debug/screenshot        — capture and return the current puzzle screenshot (cron-auth required)
  POST /solve                   — solve an arbitrary board submitted in the request body
  POST /jobs/daily-solve        — trigger the daily capture→parse→solve pipeline (cron-auth required)
  GET  /solutions/today         — fetch today's stored solution
  GET  /solutions/recent        — list recent solution summaries
  GET  /solutions/{puzzle_date} — fetch the stored solution for a specific date
"""

from contextlib import asynccontextmanager
import os
import logging
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from apps.api.board_capture import capture_google_board_screenshot, remote_browser_diagnostics
from apps.api.daily_solve import run_daily_solve, utc_puzzle_date
from apps.api.solution_storage import (
    StorageError,
    close_pool,
    get_solution,
    list_recent_solutions,
)
from apps.api.puzzle_titles import clean_puzzle_title, title_from_past_days
from solver.service import SolverError, solve_board


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

Cell = Literal["", "X", "O", "U", "B"]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SolveRequest(BaseModel):
    board: list[list[Cell]] = Field(
        ...,
        description=(
            "Tic Tac Go board. May be rectangular or ragged; omitted cells in "
            "short rows are normalized to barriers. Must contain exactly one U."
        ),
    )
    max_states: int | None = Field(
        default=None,
        ge=1,
        description="Optional search cap for bounded API calls.",
    )


class SolveStep(BaseModel):
    move: str
    board: list[list[Cell]]


class SolveResponse(BaseModel):
    solved: bool
    solver_name: str
    moves: str | None
    states_checked: int
    elapsed_ms: float
    start_board: list[list[Cell]]
    final_board: list[list[Cell]] | None
    steps: list[SolveStep]


class SolutionRecord(BaseModel):
    puzzle_date: date
    source_url: str
    parser_name: str
    solver_name: str
    board: list[list[Cell]] | None
    moves: str | None
    final_board: list[list[Cell]] | None
    step_boards: list[SolveStep]
    states_checked: int | None
    elapsed_ms: float | None
    status: Literal["pending", "solved", "unsolved", "failed"]
    error_message: str | None
    puzzle_title: str | None = None


class SolutionSummary(BaseModel):
    puzzle_date: date
    status: Literal["pending", "solved", "unsolved", "failed"]
    puzzle_title: str | None = None


class JobResponse(BaseModel):
    status: Literal["solved", "unsolved", "failed"]
    puzzle_date: date
    parser_name: str
    solver_name: str
    states_checked: int | None
    elapsed_ms: float | None
    error_message: str | None


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def require_cron_secret(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency that gates cron/admin endpoints behind a shared secret.

    The secret is read from the CRON_SECRET env var and must be supplied by
    callers in an `Authorization: Bearer <secret>` header — the same format
    that Vercel uses when it invokes cron routes, so no extra signing logic
    is needed on either side.
    """
    expected = os.getenv("CRON_SECRET")
    if not expected:
        raise HTTPException(status_code=500, detail="CRON_SECRET is not configured.")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def pending_solution(puzzle_date: date) -> SolutionRecord:
    """Return a synthetic 'pending' record for dates with no stored solution yet.

    Used as a graceful fallback so GET /solutions/{date} always returns a
    well-formed SolutionRecord rather than a 404, even before the daily job
    has run for that date.
    """
    return SolutionRecord(
        puzzle_date=puzzle_date,
        source_url=os.getenv("GOOGLE_TIC_TAC_GO_URL", ""),
        parser_name="gemini",
        solver_name="bfs",
        board=None,
        moves=None,
        final_board=None,
        step_boards=[],
        states_checked=None,
        elapsed_ms=None,
        status="pending",
        error_message="No solution has been generated for this date yet.",
        puzzle_title=title_from_past_days(puzzle_date),
    )


def with_title_fallback(record: dict) -> dict:
    """Clean a DB title and fall back to the historical manifest when unusable."""
    title = clean_puzzle_title(record.get("puzzle_title"))
    if title:
        if title == record.get("puzzle_title"):
            return record
        return {**record, "puzzle_title": title}
    return {
        **record,
        "puzzle_title": title_from_past_days(record.get("puzzle_date")),
    }


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Release process-local resources when a Cloud Run instance stops."""
    try:
        yield
    finally:
        close_pool()


app = FastAPI(title="Tic Tac Go Solver API", lifespan=lifespan)
logger = logging.getLogger("tic_tac_go.api")

PUBLIC_CACHE_CONTROL = "public, max-age=60, s-maxage=300, stale-while-revalidate=300"


def enable_public_cache(response: Response) -> None:
    """Allow browsers and shared proxies to briefly cache public solution reads."""
    response.headers["Cache-Control"] = PUBLIC_CACHE_CONTROL

allowed_origins = [
    origin.strip()
    for origin in os.getenv("API_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Debug endpoints (cron-auth required)
# ---------------------------------------------------------------------------

@app.get(
    "/debug/remote-browser",
    dependencies=[Depends(require_cron_secret)],
)
def debug_remote_browser() -> dict[str, object]:
    """Return diagnostic information about the remote browser environment."""
    return remote_browser_diagnostics()


@app.get(
    "/debug/screenshot",
    dependencies=[Depends(require_cron_secret)],
)
def debug_screenshot() -> FileResponse:
    """Capture the current puzzle and return the screenshot as a PNG."""
    try:
        result = capture_google_board_screenshot()
    except Exception as exc:
        logger.exception("debug_screenshot.failed")
        raise HTTPException(status_code=500, detail=f"Screenshot capture failed: {exc}") from exc

    return FileResponse(
        result.screenshot_path,
        media_type="image/png",
        filename="google-tic-tac-go.png",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Solver endpoint (not public)
# ---------------------------------------------------------------------------

@app.post("/solve", 
          response_model=SolveResponse,
          dependencies=[Depends(require_cron_secret)],)
def solve(request: SolveRequest) -> SolveResponse:
    """Solve a board submitted directly in the request body."""
    try:
        result = solve_board(request.board, max_states=request.max_states)
    except SolverError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SolveResponse(**result)


# ---------------------------------------------------------------------------
# Cron job endpoint (cron-auth required)
# ---------------------------------------------------------------------------

@app.post(
    "/jobs/daily-solve",
    response_model=JobResponse,
    dependencies=[Depends(require_cron_secret)],
)
def daily_solve_job() -> JobResponse:
    """Trigger the daily capture → parse → solve pipeline for today's puzzle."""
    try:
        record = run_daily_solve()
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("daily_solve_job.unhandled")
        raise HTTPException(status_code=500, detail=f"Daily solve failed: {exc}") from exc

    return JobResponse(**record)


# ---------------------------------------------------------------------------
# Solution read endpoints (public)
# ---------------------------------------------------------------------------

@app.get("/solutions/today", response_model=SolutionRecord)
def today_solution(response: Response) -> SolutionRecord:
    """Return today's stored solution, or a pending record if it hasn't run yet."""
    enable_public_cache(response)
    puzzle_date = utc_puzzle_date()
    try:
        record = get_solution(puzzle_date)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        return pending_solution(puzzle_date)
    return SolutionRecord(**with_title_fallback(record))


@app.get("/solutions/recent", response_model=list[SolutionSummary])
def recent_solutions(response: Response, limit: int = 365) -> list[SolutionSummary]:
    """Return a summary list of recent solutions, capped at 365 entries."""
    enable_public_cache(response)
    try:
        rows = list_recent_solutions(limit=min(max(limit, 1), 365))
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [SolutionSummary(**with_title_fallback(row)) for row in rows]


@app.get("/solutions/{puzzle_date}", response_model=SolutionRecord)
def solution_by_date(puzzle_date: date, response: Response) -> SolutionRecord:
    """Return the stored solution for a specific date, or a pending record."""
    enable_public_cache(response)
    try:
        record = get_solution(puzzle_date)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        return pending_solution(puzzle_date)
    return SolutionRecord(**with_title_fallback(record))
