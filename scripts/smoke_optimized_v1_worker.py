"""Offline end-to-end check for the isolated daily publisher runtime.

The open 8x8 board deliberately exercises V4's lazy NumPy scoring path. Merely
importing the optimizer, or solving a small geometry, misses that dependency.
No catalog request, production database, or publication is performed.
"""

from datetime import date

from apps.api.puzzle_catalog import CatalogEntry
from scripts.update_optimized_v1 import solve_entry, validate_record


def smoke_test():
    level = (
        "P-------"
        + "--------" * 2
        + "---O----"
        + "--------"
        + "------O-"
        + "--------" * 2
    )
    entry = CatalogEntry(date(2000, 1, 1), "Runtime smoke test", 8, 8, level)
    record, receipt = solve_entry(entry, seconds=0.05, max_nodes=100)
    validate_record(record)
    if record["board_sha256"] != entry.board_sha256:
        raise RuntimeError("Smoke-test board identity changed")
    if not record["moves"] or len(record["moves"]) > 10:
        raise RuntimeError("Smoke test failed to recover its ten-move solution")
    return record, receipt


if __name__ == "__main__":
    record, _ = smoke_test()
    print(f"Publisher runtime verified: {len(record['moves'])} moves")
