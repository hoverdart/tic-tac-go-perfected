"""Convenience entry point for the Tic Tac Go solver.

Run `python3 solve.py --help` from the repo root instead of remembering the
historical script path under randomPythonFiles.
"""

try:
    from solver.randomPythonFiles.superTicTacGoSolver import main
except ModuleNotFoundError:
    from randomPythonFiles.superTicTacGoSolver import main


if __name__ == "__main__":
    main()
