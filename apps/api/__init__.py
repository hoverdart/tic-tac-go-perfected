"""FastAPI application package for the Tic Tac Go backend.

Primary modules:
- `main`: HTTP routes and request/response models.
- `daily_solve`: capture/parse/solve/persist job orchestration.
- `board_capture`: browser screenshot acquisition.
- `board_parser`: image-to-board parser adapter.
- `solution_storage`: daily solution persistence.
- `puzzle_titles`: live and historical title lookup.
"""
