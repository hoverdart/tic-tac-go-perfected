import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from solver.gymnasium_register import ranked_real_boards
from solver.learned_search.features import candidate_features
from solver.learned_search.linear_ranker import FEATURE_NAMES, LinearRanker
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

    def test_linear_ranker_loads_trained_artifact_shape(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ranker.json"
            payload = {
                "weights": {name: 0.1 for name in FEATURE_NAMES},
                "intercept": 0.0,
                "means": {name: 0.0 for name in FEATURE_NAMES},
                "scales": {name: 1.0 for name in FEATURE_NAMES},
                "metadata": {"trainer": "test"},
            }
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")

            ranker = LinearRanker.from_json(path)

        self.assertEqual(ranker.metadata["trainer"], "test")
        self.assertEqual(set(ranker.weights), set(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
