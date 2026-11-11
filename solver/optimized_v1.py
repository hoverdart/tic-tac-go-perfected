"""optimized-v1: verified least-keystroke optimization, independent of daily V4."""

from solver.optimized_paths import OptimizationResult, optimize

SOLVER_NAME = "optimized-v1"
__all__ = ["SOLVER_NAME", "OptimizationResult", "optimize"]

if __name__ == "__main__":
    from solver.optimized_paths.__main__ import main

    main()
