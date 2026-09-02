"""Audit and safely repair missing Tic Tac Go puzzle titles in Postgres."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from urllib.request import Request, urlopen

from apps.api.puzzle_titles import (
    official_level_label,
    title_from_official_catalog,
    title_from_past_days,
)
from apps.api.solution_storage import close_pool, get_solutions_for_dates, update_missing_titles


DEFAULT_DATES = (
    "2025-11-10",
    "2025-12-12",
    "2026-08-31",
    "2026-09-01",
    "2026-09-02",
)


def load_env_file(path: str) -> None:
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def resolve_title(puzzle_date: date) -> str:
    return (
        title_from_official_catalog(puzzle_date)
        or title_from_past_days(puzzle_date)
        or official_level_label(puzzle_date)
        or puzzle_date.isoformat()
    )


def publish_cache(url: str, dates: list[date]) -> None:
    """Ask the deployed web app to invalidate only the affected cached pages."""
    secret = os.getenv("CRON_SECRET")
    if not secret:
        raise RuntimeError("CRON_SECRET is required to publish title cache changes.")
    payload = json.dumps({"dates": [value.isoformat() for value in dates]}).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Cache publication failed with HTTP {response.status}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date", action="append", dest="dates")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--publish-url",
        help="Deployed authenticated cache-publication endpoint to call after writes.",
    )
    args = parser.parse_args()
    load_env_file(args.env_file)
    dates = [date.fromisoformat(value) for value in (args.dates or DEFAULT_DATES)]
    try:
        rows = get_solutions_for_dates(dates)
        proposed = {puzzle_date: resolve_title(puzzle_date) for puzzle_date in dates}
        missing = {
            puzzle_date: title
            for puzzle_date, title in proposed.items()
            if puzzle_date in rows and not rows[puzzle_date].get("puzzle_title")
        }
        print({
            "requested": len(dates),
            "stored": len(rows),
            "missing_titles": {key.isoformat(): value for key, value in missing.items()},
        })
        if args.audit_only or args.dry_run:
            return 0
        updated = update_missing_titles(missing)
        print({"updated": {key.isoformat(): value for key, value in updated.items()}})
        if updated and args.publish_url:
            publish_cache(args.publish_url, list(updated))
            print({"cache_published": [key.isoformat() for key in updated]})
        return 0
    finally:
        close_pool()


if __name__ == "__main__":
    raise SystemExit(main())
