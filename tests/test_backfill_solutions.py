from argparse import Namespace
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from backfill_solutions import (
    ALL_PAST_DAYS,
    audit_entries,
    board_from_entry,
    build_record,
    parse_entry_date,
    run_backfill,
    is_complete_solution,
    solve_worker,
    update_solution_corpus,
)


class ResultQueue:
    def __init__(self):
        self.value = None

    def put(self, value):
        self.value = value


def backfill_args(**overrides):
    values = {
        "env_file": None,
        "solver": "push",
        "mode": "hybrid",
        "timeout_seconds": 30.0,
        "max_nodes": 500_000,
        "start_date": None,
        "end_date": None,
        "include_future": False,
        "include_missing": False,
        "limit": None,
        "audit_only": False,
        "sync_corpus_only": False,
        "list_only": False,
        "dry_run": False,
        "failure_log": None,
        "solution_corpus": None,
    }
    values.update(overrides)
    return Namespace(**values)


class BackfillSolutionsTest(unittest.TestCase):
    def test_parse_entry_date_prefers_manifest_date(self):
        self.assertEqual(parse_entry_date(ALL_PAST_DAYS[0]), date(2025, 1, 1))

    def test_board_from_entry_maps_manifest_cells(self):
        board = board_from_entry(ALL_PAST_DAYS[0])

        self.assertEqual(len(board), 8)
        self.assertEqual(len(board[0]), 8)
        self.assertEqual(board[3][6], "O")
        self.assertEqual(board[4][2], "U")
        self.assertEqual(board[0][0], "B")
        self.assertEqual(board[3][3], "")

    def test_build_record_for_timeout_stores_failed_row_shape(self):
        board = board_from_entry(ALL_PAST_DAYS[0])
        record = build_record(
            ALL_PAST_DAYS[0],
            date(2025, 1, 1),
            board,
            {
                "ok": False,
                "timed_out": True,
                "error_message": "Optimized solver exceeded 60.0 seconds.",
                "elapsed_ms": 60_000,
            },
            "hybrid",
        )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["parser_name"], "backfill_solutions")
        self.assertEqual(record["solver_name"], "optimized-hybrid")
        self.assertEqual(record["puzzle_title"], "Tutorial")
        self.assertEqual(record["board"], board)

    def test_push_worker_returns_verified_push_result_shape(self):
        queue = ResultQueue()

        solve_worker(
            [["U", "O", "O"]],
            "push",
            "hybrid",
            100,
            1.0,
            queue,
        )

        self.assertTrue(queue.value["ok"])
        self.assertTrue(queue.value["solved"])
        self.assertEqual(queue.value["solver_name"], "push-v2")

    def test_solution_corpus_only_fills_missing_paths(self):
        with TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "solutions.jsonl"
            corpus_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"id": "20250101", "solution": "RR"},
                        {"id": "20250102", "solution": None},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            updated = update_solution_corpus(
                corpus_path,
                [
                    ({"id": "20250101"}, {"moves": "LL"}),
                    (
                        {
                            "id": "20250102",
                            "width": 3,
                            "height": 1,
                            "puzzle": "POO",
                        },
                        {"moves": ""},
                    ),
                ],
            )
            rows = [
                json.loads(line)
                for line in corpus_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(updated, 1)
        self.assertEqual(rows[0]["solution"], "RR")
        self.assertEqual(rows[1]["solution"], "")

    def test_solution_corpus_skips_path_for_a_different_board(self):
        with TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "solutions.jsonl"
            corpus_path.write_text(
                json.dumps({"id": "20250102", "solution": None}) + "\n",
                encoding="utf-8",
            )

            updated = update_solution_corpus(
                corpus_path,
                [
                    (
                        {
                            "id": "20250102",
                            "width": 3,
                            "height": 1,
                            "puzzle": "POO",
                        },
                        {"moves": "R"},
                    )
                ],
            )
            row = json.loads(corpus_path.read_text(encoding="utf-8"))

        self.assertEqual(updated, 0)
        self.assertIsNone(row["solution"])

    def test_complete_solution_requires_a_move_payload(self):
        self.assertTrue(is_complete_solution({"status": "solved", "moves": ""}))
        self.assertFalse(is_complete_solution({"status": "solved", "moves": None}))

    def test_audit_distinguishes_unresolved_and_missing_rows(self):
        solved_entry, unresolved_entry, missing_entry = ALL_PAST_DAYS[:3]
        records = {
            parse_entry_date(solved_entry): {"status": "solved", "moves": "LR"},
            parse_entry_date(unresolved_entry): {"status": "failed", "moves": None},
        }

        counts = audit_entries(
            [solved_entry, unresolved_entry, missing_entry],
            records,
            log_details=False,
        )

        self.assertEqual(
            counts,
            {"manifest": 3, "stored": 2, "solved": 1, "unresolved": 1, "missing": 1},
        )

    def test_backfill_skips_solved_and_missing_rows(self):
        solved_entry, unresolved_entry, missing_entry = ALL_PAST_DAYS[:3]
        solved_date = parse_entry_date(solved_entry)
        unresolved_date = parse_entry_date(unresolved_entry)
        solved_result = {
            "ok": True,
            "solved": True,
            "moves": "",
            "states_checked": 1,
            "elapsed_ms": 1.0,
            "start_board": [["U", "O", "O"]],
            "final_board": [["U", "O", "O"]],
            "steps": [],
            "solver_name": "push-v2",
            "strategy": "precheck",
            "attempts": [],
            "failure_reason": None,
        }
        args = backfill_args()

        with TemporaryDirectory() as temp_dir:
            args.failure_log = str(Path(temp_dir) / "failures.jsonl")
            with (
                patch(
                    "backfill_solutions.iter_entries",
                    return_value=[solved_entry, unresolved_entry, missing_entry],
                ),
                patch(
                    "backfill_solutions.get_solutions_for_dates",
                    return_value={
                        solved_date: {
                            "puzzle_date": solved_date,
                            "status": "solved",
                            "moves": "LR",
                        },
                        unresolved_date: {
                            "puzzle_date": unresolved_date,
                            "status": "unsolved",
                            "moves": None,
                        },
                    },
                ),
                patch(
                    "backfill_solutions.get_solution",
                    return_value={
                        "puzzle_date": unresolved_date,
                        "status": "unsolved",
                        "moves": None,
                    },
                ),
                patch(
                    "backfill_solutions.solve_with_timeout",
                    return_value=solved_result,
                ) as solve,
                patch(
                    "backfill_solutions.upsert_solution",
                    return_value={"puzzle_date": unresolved_date, "status": "solved"},
                ) as upsert,
            ):
                result = run_backfill(args)

        self.assertEqual(result, 0)
        solve.assert_called_once()
        upsert.assert_called_once()

    def test_backfill_does_not_overwrite_row_solved_during_search(self):
        entry = ALL_PAST_DAYS[1]
        puzzle_date = parse_entry_date(entry)
        failed_result = {
            "ok": True,
            "solved": False,
            "moves": None,
            "states_checked": 500_000,
            "elapsed_ms": 30_000.0,
            "start_board": board_from_entry(entry),
            "final_board": None,
            "steps": [],
            "solver_name": "push-v2",
            "strategy": None,
            "attempts": [],
            "failure_reason": "node_limit",
        }

        with TemporaryDirectory() as temp_dir:
            failure_log = Path(temp_dir) / "failures.jsonl"
            args = backfill_args(failure_log=str(failure_log))
            with (
                patch("backfill_solutions.iter_entries", return_value=[entry]),
                patch(
                    "backfill_solutions.get_solutions_for_dates",
                    return_value={
                        puzzle_date: {
                            "puzzle_date": puzzle_date,
                            "status": "unsolved",
                            "moves": None,
                        }
                    },
                ),
                patch(
                    "backfill_solutions.get_solution",
                    return_value={
                        "puzzle_date": puzzle_date,
                        "status": "solved",
                        "moves": "LR",
                    },
                ),
                patch("backfill_solutions.clear_solution_cache") as clear_cache,
                patch(
                    "backfill_solutions.solve_with_timeout",
                    return_value=failed_result,
                ),
                patch("backfill_solutions.upsert_solution") as upsert,
            ):
                result = run_backfill(args)

            failure_rows = failure_log.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result, 0)
        clear_cache.assert_called_once_with()
        upsert.assert_not_called()
        self.assertEqual(failure_rows, [])


if __name__ == "__main__":
    unittest.main()
