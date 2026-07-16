"""Inference-only NumPy implementation of the production CNN policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


WEIGHT_SHAPES = {
    "conv1_weight": (32, 5, 3, 3),
    "conv1_bias": (32,),
    "conv2_weight": (64, 32, 3, 3),
    "conv2_bias": (64,),
    "conv3_weight": (64, 64, 3, 3),
    "conv3_bias": (64,),
    "fc1_weight": (256, 4096),
    "fc1_bias": (256,),
    "fc2_weight": (4, 256),
    "fc2_bias": (4,),
}


def _conv2d_same(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Apply a stride-one 3x3 convolution with one-cell zero padding."""
    padded = np.pad(inputs, ((0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (3, 3),
        axis=(1, 2),
    )
    height, width = inputs.shape[1:]
    patches = windows.transpose(1, 2, 0, 3, 4).reshape(height * width, -1)
    convolved = patches @ weights.reshape(weights.shape[0], -1).T
    return convolved.T.reshape(weights.shape[0], height, width) + bias[:, None, None]


class NumpySmallCNN:
    """SmallCNN-compatible policy inference without the PyTorch runtime."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = weights

    @classmethod
    def load(cls, path: str | Path) -> "NumpySmallCNN":
        with np.load(Path(path), allow_pickle=False) as stored:
            weights = {}
            for name, expected_shape in WEIGHT_SHAPES.items():
                if name not in stored:
                    raise ValueError(f"CNN checkpoint is missing {name}.")
                value = np.asarray(stored[name], dtype=np.float32)
                if value.shape != expected_shape:
                    raise ValueError(
                        f"CNN checkpoint {name} has shape {value.shape}; "
                        f"expected {expected_shape}."
                    )
                weights[name] = value.copy()
        return cls(weights)

    @staticmethod
    def get_obs(board: list[str]) -> np.ndarray:
        mapping = {".": 0, "X": 1, "O": 2, "U": 3, "B": 4}
        observation = np.zeros((5, 8, 8), dtype=np.float32)
        for row_index, row_text in enumerate(board):
            for col_index, cell in enumerate(row_text.split()):
                observation[mapping[cell], row_index, col_index] = 1.0
        return observation

    def action_scores(self, board: list[str]) -> np.ndarray:
        """Return U/D/L/R logits for one board."""
        weights = self.weights
        values = self.get_obs(board)
        values = np.maximum(
            _conv2d_same(values, weights["conv1_weight"], weights["conv1_bias"]),
            0.0,
        )
        values = np.maximum(
            _conv2d_same(values, weights["conv2_weight"], weights["conv2_bias"]),
            0.0,
        )
        values = np.maximum(
            _conv2d_same(values, weights["conv3_weight"], weights["conv3_bias"]),
            0.0,
        )
        values = np.maximum(
            weights["fc1_weight"] @ values.reshape(-1) + weights["fc1_bias"],
            0.0,
        )
        return weights["fc2_weight"] @ values + weights["fc2_bias"]
