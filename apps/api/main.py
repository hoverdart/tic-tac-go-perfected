import os
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apps.api.daily_job import run_daily_solve, utc_puzzle_date
from apps.api.storage import StorageError, get_solution
from solver.service import SolverError, solve_board


Cell = Literal["", "X", "O", "U", "B"]


class SolveRequest(BaseModel):
    board: list[list[Cell]] = Field(
        ...,
        description="Rectangular Tic Tac Go board. Must contain exactly one U.",
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


class JobResponse(BaseModel):
    status: Literal["solved", "unsolved", "failed"]
    puzzle_date: date
    parser_name: str
    solver_name: str
    states_checked: int | None
    elapsed_ms: float | None
    error_message: str | None


def require_cron_secret(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("CRON_SECRET")
    if not expected:
        raise HTTPException(status_code=500, detail="CRON_SECRET is not configured.")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def pending_solution(puzzle_date: date) -> SolutionRecord:
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
    )


app = FastAPI(title="Tic Tac Go Solver API")

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    try:
        result = solve_board(request.board, max_states=request.max_states)
    except SolverError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SolveResponse(**result)


@app.post(
    "/jobs/daily-solve",
    response_model=JobResponse,
    dependencies=[Depends(require_cron_secret)],
)
def daily_solve_job() -> JobResponse:
    try:
        record = run_daily_solve()
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JobResponse(**record)


@app.get("/solutions/today", response_model=SolutionRecord)
def today_solution() -> SolutionRecord:
    puzzle_date = utc_puzzle_date()
    try:
        record = get_solution(puzzle_date)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        return pending_solution(puzzle_date)
    return SolutionRecord(**record)


@app.get("/solutions/{puzzle_date}", response_model=SolutionRecord)
def solution_by_date(puzzle_date: date) -> SolutionRecord:
    try:
        record = get_solution(puzzle_date)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        return pending_solution(puzzle_date)
    return SolutionRecord(**record)
