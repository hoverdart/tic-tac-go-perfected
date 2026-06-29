# Tic Tac Go Perfected

Fast Tic Tac Go solvers plus a Gymnasium environment for model training.

The solver stack searches Tic Tac Go boards and reports the move string, final
board, states checked, elapsed time, and the solver used. Screenshot parsing is
currently best run with the Gemini fallback parser while the OpenCV parser is
still being worked on.

## Project Layout

- `solve.py`: easiest way to run the legacy command-line solver
- `service.py`: API-facing router that chooses the solver for each board
- `board_utils.py`: shared board normalization and JSON conversion helpers
- `heuristic_cnn_solver.py`: production wrapper for heuristic beam search with
  CNN fallback
- `beam_search.py`: beam search implementation used by the heuristic-CNN solver
- `small_cnn.py`: small behavior-cloned CNN policy used as beam guidance
- `small_cnn_policy.pt`: trained CNN checkpoint used by `heuristic_cnn_solver.py`
- `optimized_solver.py`: compact-state solver used by the API when
  `SOLVER_IMPL=optimized` on smaller boards
- `learned_search/`: experimental learned child-path ranking scaffold
- `benchmark_solvers.py`: compares legacy and optimized solver performance
- `algorithms/README.md`: notes on how each solver works and how to compare them
- `screenshots/`: drop screenshots here for the `reg-settings` command
- `legacy_solver.py`: legacy solver implementation
- `boardParsers/fallbackBoardParser.py`: Gemini screenshot parser
- `boardParsers/openCVBoardParser.py`: experimental OpenCV parser
- `gymnasium_register/`: local Gymnasium environment, board data, training
  scripts, and offline test utilities
- `requirements.txt`: minimal install for solving with Gemini
- `requirements-opencv.txt`: optional OpenCV parser dependencies
- `requirements-autoplay.txt`: optional autoplay dependency
- `requirements-training.txt`: optional DQN training dependencies

## Board Format

Boards use strings in a 2D array:

- `""` means empty
- `"X"` means an X piece
- `"O"` means an O piece
- `"U"` means the user/player piece
- `"B"` means blocked square

API and solver inputs may be rectangular or ragged. `solver.board_utils`
normalizes them into a rectangular immutable board before search; explicitly
provided empty cells remain empty, while omitted cells at the end of shorter
rows are filled with `"B"` barriers. For CNN training and Gymnasium data, the
standard board representation is `8x8`; if the visible puzzle is smaller,
unused squares to the right and bottom are filled with `"B"` so the observation
shape stays constant.

Example `3x3` puzzle padded to `8x8`:

```python
board = (("", "", "", "B", "B", "B", "B", "B"),
         ("U", "", "O", "B", "B", "B", "B", "B"),
         ("", "", "O", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))
```

## Quick Start

From the `solver/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the repo root for Gemini screenshot parsing:

```bash
GEMINI_API_KEY=your_key_here
```

## Get The Fast Solver Time

Run the solver on the default board:

```bash
python3 solve.py --quiet-progress
```

The important line is printed at the end:

```text
Time Taken: 0:00:00.123456
```

That is the fast Tic Tac Go solve time for that board.

For a bounded test run, use `--max-states`:

```bash
python3 solve.py --quiet-progress --max-states 10000
```

## Try The Optimized Solver

The API routes boards `6x6` and larger to the heuristic-CNN beam solver. Smaller
boards default to the legacy solver. Set these before starting FastAPI to use
the optimized solver for smaller boards:

```bash
SOLVER_IMPL=optimized
SOLVER_MODE=hybrid
```

`SOLVER_MODE` accepts `hybrid`, `fast`, or `exact`. `hybrid` returns a strong
solution quickly and keeps looking for a shorter one inside the state budget.
To compare both solvers on ranked boards:

```bash
python3 -m solver.benchmark_solvers --groups five six seven --limit 3
```

## Heuristic-CNN Beam Solver

File: `heuristic_cnn_solver.py`

This is the production path for larger boards. It first runs pure heuristic beam
search. If that fails, it loads `small_cnn_policy.pt` and reruns beam search with
CNN action logits mixed into the heuristic score.

Current production settings:

- beam width: `5000`
- max depth: `200`
- restarts: `5`
- random tiebreak noise: `0.05`
- random prefix steps: `[0, 0, 0, 5, 10]`
- CNN restart weights: `[0.1, 0.5, 1.0]`
- timeout: `300` seconds for pure heuristic, then another `300` seconds for
  CNN+heuristic fallback

`solve_board()` returns `solver_name` so API records show whether a board used
`bfs`, `heuristic-CNN`, or an optimized mode.

## Learned Search Scaffold

Files: `learned_search/`

This is the experimental path for training our model. It is not wired into the
API by default. The package separates the workflow into small pieces:

- `features.py`: turns one parent state and one legal child path into numeric
  model features.
- `training_data.py`: solves boards with `optimized_solver`, follows the expert
  path, and labels which candidate child was chosen at each state.
- `linear_ranker.py`: tiny runtime ranker interface and placeholder weights.
- `solver.py`: weighted A* with an extra learned child-ranking term.
- `export_training_data.py`: CLI for writing JSONL training rows.

Export a small starter dataset:

```bash
python3 -m solver.learned_search.export_training_data \
  /tmp/tic-tac-go-learned-rows.jsonl \
  --groups five \
  --limit 10
```

Each JSONL row describes one candidate child path from a parent board. `label=1`
means the child was on the optimized solver's expert path; `label=0` means it
was a legal alternative. A later training script can fit logistic regression,
linear regression, random forest, or gradient boosting against these rows and
export weights or a small model artifact.

## Solve From A Screenshot With Gemini

The easiest screenshot workflow is:

1. Put a screenshot in `screenshots/`.
2. Run:

```bash
python3 solve.py reg-settings
```

`reg-settings` chooses the newest image in `screenshots/`, uses the Gemini
parser, and runs with quiet progress.

Use `--gemini-only` to skip OpenCV and parse the screenshot directly with
Gemini:

```bash
python3 solve.py \
  --screenshot path/to/screenshot.png \
  --gemini-only \
  --quiet-progress
```

The Gemini parser returns an `8x8` board. Smaller visible boards are padded with
`"B"` blocked squares automatically.

You can also test just the Gemini parser:

```bash
python3 boardParsers/fallbackBoardParser.py path/to/screenshot.png
```

## Autoplay

After solving, the script can open Google Tic Tac Go and press the solution
moves with `pyautogui`:

```bash
python3 -m pip install -r requirements-autoplay.txt
python3 solve.py --gemini-only --screenshot path/to/screenshot.png --autoplay
```

Autoplay depends on screen position and browser layout, so use it only after
confirming the printed moves look right.

## Train The DQN Environment

Training uses PyTorch, which may not support brand-new Python versions yet. If
`pip` cannot find `torch`, create the training virtualenv with Python 3.11 or
3.12 instead of Python 3.14.

Install training dependencies:

```bash
python3 -m pip install -r gymnasium_register/requirements-training.txt
```

Run DQN training:

```bash
python3 gymnasium_register/train_DQN.py
```

Training expects the constant `8x8` padded board format described above.
