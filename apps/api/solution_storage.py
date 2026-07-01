"""
Postgres persistence layer for daily puzzle solutions.

All reads and writes go through a single `daily_solutions` table. The public
API is `upsert_solution`, `get_solution`, and `list_recent_solutions`; the
rest is internal plumbing.

The connection pool is created lazily so solver-only tools can import this
module without configuring Postgres. Schema migrations are intentionally
separate from request-time reads and writes.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import logging
import os
from threading import Lock
import time
from typing import Any


class StorageError(RuntimeError):
    """Raised when solution storage is not configured or unavailable."""


logger = logging.getLogger("tic_tac_go.daily_solve")

_pool: Any | None = None
_pool_lock = Lock()

_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
_cache_lock = Lock()
_CACHE_MAX_ENTRIES = 1024
_CACHE_MISS = object()


def _database_url() -> str:
    """Read and return the DATABASE_URL env var, or raise StorageError."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise StorageError("DATABASE_URL is not configured.")
    return database_url


def _positive_int_env(name: str, default: int) -> int:
    """Return a positive integer environment setting or its default."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float_env(name: str, default: float) -> float:
    """Return a positive floating-point environment setting or its default."""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def open_pool():
    """Create and open the process-local Postgres pool if needed."""
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise StorageError(
                "Missing dependency: install Postgres support with "
                "`python3 -m pip install 'psycopg[binary,pool]'`."
            ) from exc

        timeout = _positive_float_env("DB_POOL_TIMEOUT_SECONDS", 10.0)
        pool = ConnectionPool(
            conninfo=_database_url(),
            min_size=0,
            max_size=_positive_int_env("DB_POOL_MAX_SIZE", 5),
            timeout=timeout,
            max_idle=_positive_float_env("DB_POOL_MAX_IDLE_SECONDS", 60.0),
            kwargs={"row_factory": dict_row},
            open=False,
            name="daily-solutions",
        )
        pool.open()
        _pool = pool
        logger.info("storage.pool.opened max_size=%s", pool.max_size)
        return _pool


def close_pool() -> None:
    """Close the process-local pool during API shutdown."""
    global _pool
    with _pool_lock:
        pool = _pool
        _pool = None
    if pool is not None:
        pool.close()
        logger.info("storage.pool.closed")


def _connect():
    """Borrow one dict-row connection from the process-local pool."""
    return open_pool().connection()


def _cache_ttl_seconds() -> float:
    """Return the configured solution-read cache lifetime."""
    try:
        return max(0.0, float(os.getenv("SOLUTION_CACHE_TTL_SECONDS", "300")))
    except ValueError:
        return 300.0


def _cache_get(key: tuple[Any, ...]) -> Any:
    """Return a defensive copy of a live cached value or _CACHE_MISS."""
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return _CACHE_MISS

    now = time.monotonic()
    with _cache_lock:
        item = _cache.get(key)
        if item is None:
            return _CACHE_MISS
        expires_at, value = item
        if expires_at <= now:
            _cache.pop(key, None)
            return _CACHE_MISS
        return deepcopy(value)


def _cache_set(key: tuple[Any, ...], value: Any) -> None:
    """Cache a defensive copy while keeping memory usage bounded."""
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return

    now = time.monotonic()
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            expired = [key for key, (expires_at, _) in _cache.items() if expires_at <= now]
            for expired_key in expired:
                _cache.pop(expired_key, None)
            if len(_cache) >= _CACHE_MAX_ENTRIES:
                _cache.clear()
        _cache[key] = (now + ttl, deepcopy(value))


def clear_solution_cache() -> None:
    """Invalidate cached reads after a database write."""
    with _cache_lock:
        _cache.clear()


def _json(value: Any):
    """Wrap a Python value in a psycopg Jsonb object."""
    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise StorageError(
            "Missing dependency: install Postgres support with "
            "`python3 -m pip install 'psycopg[binary,pool]'`."
        ) from exc

    return Jsonb(value)


def run_migrations() -> None:
    """Apply idempotent schema changes outside normal request processing."""
    logger.info("storage.migrate.start")
    with _connect() as conn:
        # Serialize concurrent Cloud Run instance startups for this schema.
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (84736291,))
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
    logger.info("storage.migrate.done")


def init_db() -> None:
    """Compatibility alias for older scripts; prefer run_migrations()."""
    run_migrations()


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

    clear_solution_cache()
    logger.info(
        "storage.upsert.done puzzle_date=%s status=%s updated_at=%s",
        stored.get("puzzle_date"),
        stored.get("status"),
        stored.get("updated_at"),
    )
    return stored


def get_solution(puzzle_date: date) -> dict[str, Any] | None:
    """Return the stored solution for a given date, or None if it doesn't exist."""
    cache_key = ("solution", puzzle_date)
    cached = _cache_get(cache_key)
    if cached is not _CACHE_MISS:
        logger.info("storage.get.cache_hit puzzle_date=%s", puzzle_date)
        return cached

    logger.info("storage.get.start puzzle_date=%s", puzzle_date)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM daily_solutions WHERE puzzle_date = %s",
            (puzzle_date,),
        ).fetchone()
        result = dict(row) if row else None
    _cache_set(cache_key, result)
    logger.info("storage.get.done puzzle_date=%s found=%s", puzzle_date, row is not None)
    return result


def list_recent_solutions(limit: int = 30) -> list[dict[str, Any]]:
    """Return a lightweight summary list of the most recent solutions."""
    cache_key = ("recent", limit)
    cached = _cache_get(cache_key)
    if cached is not _CACHE_MISS:
        logger.info("storage.list.cache_hit limit=%s", limit)
        return cached

    logger.info("storage.list.start limit=%s", limit)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT puzzle_date, status, puzzle_title FROM daily_solutions ORDER BY puzzle_date DESC LIMIT %s",
            (limit,),
        ).fetchall()
        result = [dict(row) for row in rows]
    _cache_set(cache_key, result)
    logger.info("storage.list.done count=%s", len(result))
    return result
