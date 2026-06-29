"""Compatibility wrapper for the renamed `solver.legacy_solver` module."""

try:
    from solver.legacy_solver import *  # noqa: F401,F403
except ModuleNotFoundError:
    from legacy_solver import *  # type: ignore # noqa: F401,F403
