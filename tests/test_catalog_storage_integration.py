"""Opt-in real Postgres coverage; creates and removes its own isolated schema."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from apps.api import solution_storage as storage


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL for isolated Postgres integration tests")
class CatalogStorageIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo
        cls.admin = psycopg.connect(os.environ["TEST_DATABASE_URL"], autocommit=True)
        cls.schema = "catalog_test_" + uuid4().hex
        cls.admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.environment = patch.dict(os.environ, {"DATABASE_URL": make_conninfo(
            os.environ["TEST_DATABASE_URL"], options=f"-c search_path={cls.schema}"),
            "SOLUTION_CACHE_TTL_SECONDS": "0"})
        cls.environment.start()
        storage.close_pool()
        storage.run_migrations()

    @classmethod
    def tearDownClass(cls):
        from psycopg import sql
        storage.close_pool()
        cls.environment.stop()
        cls.admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))
        cls.admin.close()

    def setUp(self):
        with storage._connect() as conn:
            conn.execute("TRUNCATE daily_solutions")

    def record(self, status="solved"):
        return {"puzzle_date": date(2026, 10, 21), "source_url": "https://www.gstatic.com/test-bundle",
                "parser_name": "catalog", "solver_name": "push-v3", "board": [["U", "O", "O"]],
                "moves": "" if status == "solved" else None, "final_board": [["U", "O", "O"]] if status == "solved" else None,
                "step_boards": [], "states_checked": 1, "elapsed_ms": 1, "status": status,
                "error_message": None if status == "solved" else "timeout", "puzzle_title": "Test"}

    def test_failed_retry_preserves_every_solved_field(self):
        original = storage.upsert_solution(self.record(), preserve_solved=True)
        for status in ["failed", "unsolved"]:
            failed = self.record(status)
            failed.update(board=None, parser_name="not-run", source_url="https://example.test/failure")
            retained = storage.upsert_solution(failed, preserve_solved=True)
            self.assertEqual(retained, original)

    def test_failure_can_insert_then_be_replaced_by_a_solution(self):
        self.assertEqual(storage.upsert_solution(self.record("failed"), preserve_solved=True)["status"], "failed")
        self.assertEqual(storage.upsert_solution(self.record(), preserve_solved=True)["status"], "solved")
        # A verified repair may replace solved data; non-daily callers keep explicit replacement semantics.
        self.assertEqual(storage.upsert_solution(self.record("failed"))["status"], "failed")

    def test_concurrent_completions_never_downgrade_a_solved_row(self):
        records = [self.record("failed") for _ in range(12)]
        records[4] = self.record()
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda row: storage.upsert_solution(row, preserve_solved=True), records))
        self.assertEqual(storage.get_solution(date(2026, 10, 21))["status"], "solved")


if __name__ == "__main__":
    unittest.main()
