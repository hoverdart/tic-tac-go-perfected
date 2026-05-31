import os
import logging
from datetime import date
from typing import Any


class StorageError(RuntimeError):
    """Raised when solution storage is not configured or unavailable."""


logger = logging.getLogger("tic_tac_go.daily_solve")


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise StorageError("DATABASE_URL is not configured.")
    return database_url


def _connect():
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
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def init_db() -> None:
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
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    logger.info("storage.init_db.done")


def upsert_solution(record: dict[str, Any]) -> dict[str, Any]:
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
                error_message
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
                %(error_message)s
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
                updated_at = now()
            RETURNING *
            """,
            {
                **record,
                "board": _json(record.get("board")),
                "final_board": _json(record.get("final_board")),
                "step_boards": _json(record.get("step_boards", [])),
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
    logger.info("storage.get.start puzzle_date=%s", puzzle_date)
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM daily_solutions WHERE puzzle_date = %s",
            (puzzle_date,),
        ).fetchone()
        logger.info("storage.get.done puzzle_date=%s found=%s", puzzle_date, row is not None)
        return dict(row) if row else None
