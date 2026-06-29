"""Convenience entry point for the Tic Tac Go solver.

Run `python3 solve.py --help` from the repo root instead of remembering the
legacy solver module path.
"""

try:
    from solver.legacy_solver import main
except ModuleNotFoundError:
    from legacy_solver import main


if __name__ == "__main__":
    main()
