"""Reusable Tic Tac Go solver package.

Entry points:
- `service`: API-facing solver router.
- `optimized_solver`: weighted A* compact-state solver.
- `heuristic_cnn_solver`: heuristic beam search with CNN fallback.
- `learned_search`: experimental learned child-path ranking scaffold.
- `legacy_solver`: legacy search implementation and CLI helpers.
- `board_utils`: shared board normalization helpers.
"""

from solver.service import SolverError, solve_board

__all__ = ["SolverError", "solve_board"]
