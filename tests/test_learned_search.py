import unittest

from solver.gymnasium_register import ranked_real_boards
from solver.learned_search.features import candidate_features
from solver.learned_search.solver import solve as learned_solve
from solver.learned_search.training_data import expert_rows_for_solution
from solver import optimized_solver


class LearnedSearchTest(unittest.TestCase):
    def test_candidate_features_describe_child_transition(self):
        board = ranked_real_boards.fiveBoards[0]
        start_board = optimized_solver.normalize_board(board)
        geometry = optimized_solver._geometry_for_board(start_board)
        parent_key = optimized_solver._to_key(start_board)
        child_key, segment = next(optimized_solver._next_states(parent_key, geometry))

        features = candidate_features(parent_key, child_key, segment, geometry)

        self.assertGreaterEqual(features.segment_length, 1)
        self.assertIn(features.child_solved, (0, 1))
        self.assertIn(features.child_lost, (0, 1))
        self.assertIn(features.child_pruned, (0, 1))

    def test_expert_rows_label_one_child_per_depth(self):
        board = ranked_real_boards.fiveBoards[0]
        moves, _final_board, _states = optimized_solver.solve(
            board,
            progress_every=0,
            max_states=10_000,
        )

        rows = expert_rows_for_solution(board, moves, board_id="test-board")
        labels_by_depth = {}
        for row in rows:
            labels_by_depth.setdefault(row["depth"], 0)
            labels_by_depth[row["depth"]] += row["label"]

        self.assertTrue(rows)
        self.assertTrue(all(label_count == 1 for label_count in labels_by_depth.values()))
        self.assertEqual({row["board_id"] for row in rows}, {"test-board"})

    def test_learned_solver_solves_ranked_regression_board(self):
        board = ranked_real_boards.fiveBoards[0]

        moves, final_board, states_checked = learned_solve(
            board,
            progress_every=0,
            max_states=10_000,
        )

        self.assertIsNotNone(moves)
        self.assertIsNotNone(final_board)
        self.assertGreaterEqual(states_checked, 1)


if __name__ == "__main__":
    unittest.main()
