import unittest

import numpy as np

from solver.heuristic_cnn_solver import MODEL_PATH, load_model
from solver.numpy_cnn import NumpySmallCNN


class NumpyCNNTest(unittest.TestCase):
    def test_production_checkpoint_loads_and_scores_board(self):
        model = NumpySmallCNN.load(MODEL_PATH)
        scores = model.action_scores(
            [
                "U O O B B B B B",
                "B B B B B B B B",
                "B B B B B B B B",
                "B B B B B B B B",
                "B B B B B B B B",
                "B B B B B B B B",
                "B B B B B B B B",
                "B B B B B B B B",
            ]
        )

        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(
            scores,
            [-7.779166, 5.2888536, -16.182497, 6.224556],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_wrapper_caches_numpy_model(self):
        first = load_model()
        second = load_model()

        self.assertIs(first, second)
        self.assertIsInstance(first, NumpySmallCNN)


if __name__ == "__main__":
    unittest.main()
