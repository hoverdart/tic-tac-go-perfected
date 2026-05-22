# Tic Tac Go Perfected

Fast Tic Tac Go solver plus a Gymnasium environment for DQN training.

The solver searches Tic Tac Go boards and prints the move string, final board,
states checked, and `Time Taken`. Screenshot parsing is currently best run with
the Gemini fallback parser while the OpenCV parser is still being worked on.

## Project Layout

- `solve.py`: easiest way to run the fast solver
- `screenshots/`: drop screenshots here for the `reg-settings` command
- `randomPythonFiles/superTicTacGoSolver.py`: solver implementation
- `boardParsers/fallbackBoardParser.py`: Gemini screenshot parser
- `boardParsers/openCVBoardParser.py`: experimental OpenCV parser
- `gymnasium_register/`: local Gymnasium environment and DQN training script
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

For DQN training, the standard board representation is always `8x8`. If the
visible puzzle is smaller, unused squares to the right and bottom are filled
with `"B"` so the observation shape stays constant.

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

From the repo root:

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
python3 -m pip install -r requirements-training.txt
```

Run:

```bash
python3 gymnasium_register/train.py
```

Training expects the constant `8x8` padded board format described above.
