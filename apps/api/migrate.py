"""Apply the API database schema before serving requests."""

from __future__ import annotations

import logging

from apps.api.solution_storage import close_pool, run_migrations


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    try:
        run_migrations()
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
