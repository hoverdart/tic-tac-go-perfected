import unittest
from collections import Counter

from solver.push_solver.benchmark_hard_tail import (
    HARD_TAIL_BOARD_IDS,
    classify_rows,
)


class PushSolverHardTailTest(unittest.TestCase):
    def test_hard_tail_suite_keeps_original_sixteen_boards(self):
        self.assertEqual(len(HARD_TAIL_BOARD_IDS), 16)
        self.assertEqual(len(set(HARD_TAIL_BOARD_IDS)), 16)

    def test_hard_tail_classifies_known_solution_availability(self):
        rows = classify_rows()
        statuses = Counter(row.known_solution_status for row in rows)

        self.assertEqual(len(rows), 16)
        self.assertEqual(statuses["known_solution"], 11)
        self.assertEqual(statuses["no_known_solution"], 5)

        no_known_ids = {
            row.board_id
            for row in rows
            if row.known_solution_status == "no_known_solution"
        }
        self.assertEqual(
            no_known_ids,
            {"20251207", "20251212", "20251228", "20260124", "20260314"},
        )


if __name__ == "__main__":
    unittest.main()
