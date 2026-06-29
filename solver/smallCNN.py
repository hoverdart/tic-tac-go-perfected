"""Compatibility wrapper for the renamed `solver.small_cnn` module."""

try:
    from solver.small_cnn import *  # noqa: F401,F403
except ModuleNotFoundError:
    from small_cnn import *  # type: ignore # noqa: F401,F403
