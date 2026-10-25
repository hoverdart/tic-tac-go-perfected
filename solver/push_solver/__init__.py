"""Classical push-level Tic Tac Go solver."""

from solver.push_solver.core import PushSolveResult, SearchAttempt, solve, solve_v1, solve_v2
from solver.push_solver.verify import verify_solution

# Keep the historical default stable for direct/custom callers and comparisons.
solve_v3 = solve


def solve_v4(board, **kwargs):
    from solver.push_solver.v4 import solve_v4 as implementation
    return implementation(board, **kwargs)

__all__ = [
    "PushSolveResult",
    "SearchAttempt",
    "solve",
    "solve_v1",
    "solve_v2",
    "solve_v3",
    "solve_v4",
    "verify_solution",
]
