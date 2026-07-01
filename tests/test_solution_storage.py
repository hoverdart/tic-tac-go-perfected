"""Focused tests for the solution storage read cache."""

from datetime import date
import os
import unittest
from unittest.mock import patch

from apps.api import solution_storage


class _QueryResult:
    def __init__(self, *, row=None, rows=None):
        self.row = row
        self.rows = rows or []

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, result):
        self.result = result
        self.execute_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *_args, **_kwargs):
        self.execute_count += 1
        return self.result


class SolutionStorageCacheTests(unittest.TestCase):
    def setUp(self):
        solution_storage.clear_solution_cache()

    def tearDown(self):
        solution_storage.clear_solution_cache()

    def test_get_solution_reuses_cached_result(self):
        puzzle_date = date(2026, 7, 1)
        connection = _Connection(
            _QueryResult(row={"puzzle_date": puzzle_date, "status": "complete"})
        )

        with patch.object(solution_storage, "_connect", return_value=connection):
            first = solution_storage.get_solution(puzzle_date)
            first["status"] = "mutated"
            second = solution_storage.get_solution(puzzle_date)

        self.assertEqual(connection.execute_count, 1)
        self.assertEqual(second["status"], "complete")

    def test_cache_clear_forces_next_read_to_query(self):
        puzzle_date = date(2026, 7, 1)
        connection = _Connection(
            _QueryResult(row={"puzzle_date": puzzle_date, "status": "complete"})
        )

        with patch.object(solution_storage, "_connect", return_value=connection):
            solution_storage.get_solution(puzzle_date)
            solution_storage.clear_solution_cache()
            solution_storage.get_solution(puzzle_date)

        self.assertEqual(connection.execute_count, 2)

    def test_zero_ttl_disables_cache(self):
        puzzle_date = date(2026, 7, 1)
        connection = _Connection(_QueryResult(row=None))

        with (
            patch.dict(os.environ, {"SOLUTION_CACHE_TTL_SECONDS": "0"}),
            patch.object(solution_storage, "_connect", return_value=connection),
        ):
            solution_storage.get_solution(puzzle_date)
            solution_storage.get_solution(puzzle_date)

        self.assertEqual(connection.execute_count, 2)


if __name__ == "__main__":
    unittest.main()
