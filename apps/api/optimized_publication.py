"""Import the optimizer's public, revision-pinned publication into Postgres."""

from datetime import UTC, date, datetime
from typing import Literal
from urllib.request import urlopen

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

from apps.api.solution_storage import get_optimized_publication, publish_optimized_snapshot
from solver.legacy_solver import apply_single_move, solved
from solver.push_solver.verify import verify_solution

FEED_SCHEMA = "optimized-v1-feed-v1"
FEED_URL = "https://raw.githubusercontent.com/hoverdart/tic-tac-go-perfected/{revision}/optimized-v1.json"
MAX_FEED_BYTES = 20_000_000


class OptimizedRevision(BaseModel):
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class OptimizedSolution(BaseModel):
    puzzle_date: date
    puzzle_title: str = Field(min_length=1, max_length=80)
    board_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    board: list[list[Literal["", "U", "X", "O", "B"]]] = Field(min_length=8, max_length=8)
    moves: str = Field(pattern=r"^[UDLR]*$", max_length=10_000)
    solver_name: Literal["optimized-v1"]
    optimality_proven: bool
    lower_bound: int = Field(ge=0)
    updated_at: AwareDatetime
    provenance: dict = Field(default_factory=dict)

    @field_validator("board")
    @classmethod
    def validate_board(cls, board):
        if any(len(row) != 8 for row in board) or sum(row.count("U") for row in board) != 1:
            raise ValueError("Expected an 8×8 board with one player")
        return board

    @model_validator(mode="after")
    def verify_path(self):
        if self.lower_bound > len(self.moves) or (
            self.optimality_proven and self.lower_bound != len(self.moves)
        ):
            raise ValueError("Invalid optimized path proof")
        result = verify_solution(self.board, self.moves)
        if not result.ok:
            raise ValueError(f"Optimized path failed verification: {result.error}")
        state = tuple(map(tuple, self.board))
        for move in self.moves:
            if solved(state):
                raise ValueError("Optimized path continues after the first win")
            state = apply_single_move(state, move)
        return self


class OptimizedFeed(BaseModel):
    schema_version: Literal["optimized-v1-feed-v1"] = Field(alias="schema")
    solver_name: Literal["optimized-v1"]
    solutions: list[OptimizedSolution] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_dates(self):
        dates = {row.puzzle_date for row in self.solutions}
        if len(dates) != len(self.solutions) or max(dates) > datetime.now(UTC).date():
            raise ValueError("Optimized feed contains duplicate or unreleased dates")
        return self


def sync_optimized_publication(revision: str) -> list[date]:
    current = get_optimized_publication()
    previous_revision = current["revision"] if current else None
    if previous_revision == revision:
        return current["pending_dates"]
    with urlopen(FEED_URL.format(revision=revision), timeout=15) as response:
        body = response.read(MAX_FEED_BYTES + 1)
    if len(body) > MAX_FEED_BYTES:
        raise ValueError("Optimized feed exceeds the publication size limit")
    feed = OptimizedFeed.model_validate_json(body)
    solutions = [row.model_dump(mode="json") for row in feed.solutions]
    return publish_optimized_snapshot(revision, previous_revision, solutions)
