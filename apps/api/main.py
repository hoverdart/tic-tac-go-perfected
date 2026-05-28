import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
