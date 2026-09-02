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


class _TitleBackfillConnection(_Connection):
    def __init__(self, row):
        super().__init__(_QueryResult(row=row))
        self.calls = []

    def execute(self, query, params):
        self.execute_count += 1
        self.calls.append((query, params))
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

    def test_get_solutions_for_dates_uses_one_query(self):
        first_date = date(2026, 7, 1)
        second_date = date(2026, 7, 2)
        connection = _Connection(
            _QueryResult(
                rows=[{"puzzle_date": first_date, "status": "solved"}]
            )
        )

        with patch.object(solution_storage, "_connect", return_value=connection):
            rows = solution_storage.get_solutions_for_dates(
                [first_date, second_date, first_date]
            )

        self.assertEqual(connection.execute_count, 1)
        self.assertEqual(list(rows), [first_date])

    def test_title_only_backfill_updates_only_blank_title_column(self):
        puzzle_date = date(2026, 9, 1)
        connection = _TitleBackfillConnection(
            {"puzzle_date": puzzle_date, "puzzle_title": "Level 2026-09-01"}
        )
        with patch.object(solution_storage, "_connect", return_value=connection):
            updated = solution_storage.update_missing_titles(
                {puzzle_date: "Level 2026-09-01"}
            )

        self.assertEqual(updated, {puzzle_date: "Level 2026-09-01"})
        query, params = connection.calls[0]
        self.assertIn("SET puzzle_title = %s", query)
        self.assertIn("puzzle_title IS NULL", query)
        self.assertNotIn("SET status", query)
        self.assertEqual(params, ("Level 2026-09-01", puzzle_date))


if __name__ == "__main__":
    unittest.main()
