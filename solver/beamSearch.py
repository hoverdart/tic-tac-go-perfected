"""Compatibility wrapper for the renamed `solver.beam_search` module."""

try:
    from solver.beam_search import *  # noqa: F401,F403
except ModuleNotFoundError:
    from beam_search import *  # type: ignore # noqa: F401,F403
