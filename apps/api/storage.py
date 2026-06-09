"""
Postgres persistence layer for daily puzzle solutions.

All reads and writes go through a single `daily_solutions` table. The public
API is `upsert_solution`, `get_solution`, and `list_recent_solutions`; the
rest is internal plumbing.

`psycopg` is imported lazily (inside `_connect`) rather than at module top so
the API server stays importable in environments where the Postgres driver isn't
installed — for example when running only the solver locally or during tests
that mock the storage layer.
"""

import os
import logging
from datetime import date
from typing import Any


class StorageError(RuntimeError):
    """Raised when solution storage is not configured or unavailable."""


logger = logging.getLogger("tic_tac_go.daily_solve")


def _database_url() -> str:
    """Read and return the DATABASE_URL env var, or raise StorageError."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise StorageError("DATABASE_URL is not configured.")
    return database_url


def _connect():
    """Open a psycopg connection with dict-style row access.

    psycopg is imported here rather than at module top — see module docstring.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise StorageError(
            "Missing dependency: install Postgres support with "
            "`python3 -m pip install psycopg[binary]`."
        ) from exc

    return psycopg.connect(_database_url(), row_factory=dict_row)


def _json(value: Any):
    """Wrap a Python value in a psycopg Jsonb object.

    psycopg won't automatically serialise nested Python dicts/lists into
    Postgres jsonb columns — we need to wrap them explicitly. Board state
    is stored as jsonb so it's queryable; plain text serialisation wouldn't
    be.
    """
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def init_db() -> None:
    """Ensure the daily_solutions table exists and is up to date.

    Called at the top of every public storage function rather than once at
    startup so the table is created on the first real request in a cold
    environment (e.g. a fresh Vercel deployment) without needing a separate
    migration step.

    Column additions use `ADD COLUMN IF NOT EXISTS` so they're safe to re-run
    on every call. This is intentionally lightweight — proper migrations would
    be overkill for a single-table schema that evolves slowly.
    """
    logger.info("storage.init_db.start")
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_solutions (
                puzzle_date date PRIMARY KEY,
                source_url text NOT NULL,
                parser_name text NOT NULL,
                solver_name text NOT NULL,
                board jsonb,
                moves text,
                final_board jsonb,
                step_boards jsonb NOT NULL DEFAULT '[]'::jsonb,
                states_checked integer,
                elapsed_ms double precision,
                status text NOT NULL,
                error_message text,
                puzzle_title text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        # puzzle_title was added after the initial release; existing rows are
        # migrated automatically on the next access.
        conn.execute(
            "ALTER TABLE daily_solutions ADD COLUMN IF NOT EXISTS puzzle_title text"
        )
    logger.info("storage.init_db.done")


def upsert_solution(record: dict[str, Any]) -> dict[str, Any]:
    """Insert or replace a daily solution record, returning the stored row.

    Uses Postgres INSERT ... ON CONFLICT DO UPDATE so re-running the daily job
    for the same date (e.g. after a parse failure) always reflects the latest
    result rather than creating a duplicate.
    """
    logger.info(
        "storage.upsert.start puzzle_date=%s status=%s states_checked=%s elapsed_ms=%s",
        record.get("puzzle_date"),
        record.get("status"),
        record.get("states_checked"),
        record.get("elapsed_ms"),
    )
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            INSERT INTO daily_solutions (
                puzzle_date,
                source_url,
                parser_name,
                solver_name,
                board,
                moves,
                final_board,
                step_boards,
                states_checked,
                elapsed_ms,
                status,
                error_message,
                puzzle_title
            )
            VALUES (
                %(puzzle_date)s,
                %(source_url)s,
                %(parser_name)s,
                %(solver_name)s,
                %(board)s,
                %(moves)s,
                %(final_board)s,
                %(step_boards)s,
                %(states_checked)s,
                %(elapsed_ms)s,
                %(status)s,
                %(error_message)s,
                %(puzzle_title)s
            )
            ON CONFLICT (puzzle_date) DO UPDATE SET
                source_url = EXCLUDED.source_url,
                parser_name = EXCLUDED.parser_name,
                solver_name = EXCLUDED.solver_name,
                board = EXCLUDED.board,
                moves = EXCLUDED.moves,
                final_board = EXCLUDED.final_board,
                step_boards = EXCLUDED.step_boards,
                states_checked = EXCLUDED.states_checked,
                elapsed_ms = EXCLUDED.elapsed_ms,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                puzzle_title = EXCLUDED.puzzle_title,
                updated_at = now()
            RETURNING *
            """,
            {
                **record,
                "board": _json(record.get("board")),
                "final_board": _json(record.get("final_board")),
                "step_boards": _json(record.get("step_boards", [])),
                "puzzle_title": record.get("puzzle_title"),
            },
        ).fetchone()
        stored = dict(row)
        logger.info(
            "storage.upsert.done puzzle_date=%s status=%s updated_at=%s",
            stored.get("puzzle_date"),
            stored.get("status"),
            stored.get("updated_at"),
        )
        return stored


def get_solution(puzzle_date: date) -> dict[str, Any] | None:
    """Return the stored solution for a given date, or None if it doesn't exist."""
    logger.info("storage.get.start puzzle_date=%s", puzzle_date)
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM daily_solutions WHERE puzzle_date = %s",
            (puzzle_date,),
        ).fetchone()
        logger.info("storage.get.done puzzle_date=%s found=%s", puzzle_date, row is not None)
        return dict(row) if row else None


def list_recent_solutions(limit: int = 30) -> list[dict[str, Any]]:
    """Return a lightweight summary list of the most recent solutions."""
    logger.info("storage.list.start limit=%s", limit)
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT puzzle_date, status, puzzle_title FROM daily_solutions ORDER BY puzzle_date DESC LIMIT %s",
            (limit,),
        ).fetchall()
        logger.info("storage.list.done count=%s", len(rows))
        return [dict(row) for row in rows]
