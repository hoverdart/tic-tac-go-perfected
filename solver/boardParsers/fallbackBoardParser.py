"""
Gemini Vision board parser for Tic Tac Go screenshots.

Sends a screenshot to the Gemini API, asks it to describe the visible board as a
JSON matrix, then validates and normalizes the response into a padded 8x8 board
ready for the solver.

Retry strategy: if the first Gemini response fails validation, a second call is
made with the bad response appended to the prompt so Gemini can self-correct. This
single retry handles the most common failure modes (wrong wrapper key, extra text,
slightly wrong cell symbols) without looping indefinitely.

This file can be run directly as a CLI tool (python fallbackBoardParser.py
<image_path>) as well as imported by the FastAPI server, so it includes its own
.env loading for the API key.
"""

import argparse
import json
import mimetypes
import os
from pathlib import Path


# The downstream solver and any ML models (e.g. the historical DQN) always expect
# an 8x8 board. Smaller visible boards (e.g. a 5x5 puzzle) are padded to this size
# with barrier cells so the input shape is always consistent.
STANDARD_BOARD_SIZE = 8

# Gemini may return a board as small as 3x3 for very simple puzzles. Anything
# smaller is almost certainly a parsing error.
MIN_VISIBLE_BOARD_SIZE = 3

ALLOWED_INPUT_CELLS = {"X", "O", "U", "B", ".", ""}

PROMPT = """Find the standard Google Tic Tac Go board in this screenshot.

Return only the visible playable board state as a rectangular matrix.
The visible board can be smaller than 8x8, such as 3x3 or 6x6.
Read the board top-to-bottom, left-to-right.
Use exactly these symbols:
- X for X pieces
- O for O pieces
- U for the user's movable piece
- B for barriers / blocked squares
- . for empty cells

Do not infer moves, solve the game, or change any pieces. Identify only what is visible now.
"""

# The retry prompt intentionally includes the previous bad response so Gemini can
# see exactly what it produced and correct it. This "show your work" pattern works
# better than just repeating the original prompt.
RETRY_PROMPT = """The previous response did not validate as a Tic Tac Go board.

Return only corrected JSON for the visible board:
- a rectangular board from 3x3 through 8x8
- allowed cells only: X, O, U, B, .
- exactly one U

Read top-to-bottom, left-to-right. Do not solve the board.
Previous invalid response:
"""

# JSON Schema passed to Gemini's response_json_schema parameter. Constraining the
# output format here eliminates most parsing errors before they reach our validation
# layer — Gemini will try to conform to this schema rather than free-styling JSON.
BOARD_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "board": {
            "type": "array",
            "minItems": MIN_VISIBLE_BOARD_SIZE,
            "maxItems": STANDARD_BOARD_SIZE,
            "items": {
                "type": "array",
                "minItems": MIN_VISIBLE_BOARD_SIZE,
                "maxItems": STANDARD_BOARD_SIZE,
                "items": {
                    "type": "string",
                    "enum": ["X", "O", "U", "B", "."],
                },
            },
        }
    },
    "required": ["board"],
}


class BoardParseError(ValueError):
    """Raised when a Gemini response cannot be converted into a valid board."""


def pad_board_to_standard_size(board):
    """Pad a smaller visible board out to STANDARD_BOARD_SIZE x STANDARD_BOARD_SIZE.

    Extra columns are filled with "B" (barrier) cells, and extra rows are added as
    full barrier rows. This keeps the board shape fixed at 8x8 for the solver and
    any downstream models, regardless of the actual puzzle size.
    """
    padded = []
    for row in board:
        padded.append(row + ["B"] * (STANDARD_BOARD_SIZE - len(row)))

    while len(padded) < STANDARD_BOARD_SIZE:
        padded.append(["B"] * STANDARD_BOARD_SIZE)

    return padded


def validate_board(board):
    """Validate a parsed board and return it padded to 8x8.

    Three layers of checks:
      1. Structural: must be a list of lists with consistent width and between
         MIN_VISIBLE_BOARD_SIZE and STANDARD_BOARD_SIZE rows/columns.
      2. Cell validity: every cell must be a string in ALLOWED_INPUT_CELLS.
      3. Exactly one U piece: the solver requires a unique user position.

    "." cells are normalized back to "" (the solver's internal empty representation)
    before padding. Returns the padded board on success, raises BoardParseError on
    any violation.
    """
    if not isinstance(board, list):
        raise BoardParseError("Board must be a list of rows.")
    if not MIN_VISIBLE_BOARD_SIZE <= len(board) <= STANDARD_BOARD_SIZE:
        raise BoardParseError(
            "Board must have between "
            f"{MIN_VISIBLE_BOARD_SIZE} and {STANDARD_BOARD_SIZE} rows, "
            f"got {len(board)}."
        )

    user_count = 0
    normalized = []
    visible_width = None
    for row_index, row in enumerate(board):
        if not isinstance(row, list):
            raise BoardParseError(f"Row {row_index} must be a list.")
        if not MIN_VISIBLE_BOARD_SIZE <= len(row) <= STANDARD_BOARD_SIZE:
            raise BoardParseError(
                "Rows must have between "
                f"{MIN_VISIBLE_BOARD_SIZE} and {STANDARD_BOARD_SIZE} cells, "
                f"got {len(row)} in row {row_index}."
            )
        # All rows must be the same width (rectangular board).
        if visible_width is None:
            visible_width = len(row)
        elif len(row) != visible_width:
            raise BoardParseError("Board rows must all be the same length.")

        normalized_row = []
        for col_index, cell in enumerate(row):
            if cell is None:
                cell = ""
            if not isinstance(cell, str):
                raise BoardParseError(
                    f"Cell ({row_index}, {col_index}) must be a string."
                )

            cell = cell.strip().upper()
            if cell not in ALLOWED_INPUT_CELLS:
                raise BoardParseError(f"Cell ({row_index}, {col_index}) has invalid value {cell!r}.")

            # Convert "." back to "" for internal consistency (the solver uses "" for empty).
            if cell == ".":
                cell = ""
            if cell == "U":
                user_count += 1
            normalized_row.append(cell)

        normalized.append(normalized_row)

    if user_count != 1:
        raise BoardParseError(f"Board must contain exactly one U, got {user_count}.")

    return pad_board_to_standard_size(normalized)


def board_from_response_text(response_text):
    """Parse a raw Gemini JSON response string into a validated, padded board.

    Handles two response shapes:
      - `{"board": [[...]]}` — the expected schema-wrapped format.
      - `[[...]]` — Gemini occasionally strips the outer object and returns the
        array directly; we accept both so a schema non-compliance doesn't fail silently.

    Raises BoardParseError if the JSON is invalid or the board fails validation.
    """
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise BoardParseError(f"Gemini did not return valid JSON: {exc}") from exc

    if isinstance(payload, dict):
        if "board" not in payload:
            raise BoardParseError("Gemini JSON must contain a 'board' key.")
        payload = payload["board"]

    return validate_board(payload)


def _load_genai():
    """Import and return the google-genai SDK, with a friendly error if missing."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install Gemini's SDK with "
            "`python3 -m pip install google-genai`."
        ) from exc

    return genai, types


def _mime_type_for(image_path):
    """Guess the MIME type for an image path, defaulting to image/png."""
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/png"


def _api_key_from_env_file():
    """Search for GEMINI_API_KEY in .env files near the project root.

    This exists because this parser can be invoked as a standalone CLI tool, not
    just via the FastAPI server. When run directly, environment variables from the
    server's process aren't available, so we manually scan .env files in a few
    candidate locations (cwd, this file's directory, and its parent).

    Returns the key string if found, or None if no .env file contains it.
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]

    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() == "GEMINI_API_KEY":
                return value.strip().strip('"').strip("'")

    return None


def _call_gemini(image_path, prompt, model):
    """Send an image and prompt to Gemini and return the raw response text.

    Uses response_json_schema to constrain Gemini's output to the expected board
    shape, which significantly reduces parsing errors on the first attempt.
    """
    genai, types = _load_genai()
    image_bytes = Path(image_path).read_bytes()

    # Prefer the env var (set by the server); fall back to .env file scanning for
    # standalone CLI usage.
    api_key = os.environ.get("GEMINI_API_KEY") or _api_key_from_env_file()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=_mime_type_for(image_path),
            ),
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": BOARD_RESPONSE_SCHEMA,
        },
    )
    return response.text


def parse_board_from_image(image_path, model="gemini-flash-latest"):
    """Parse a Tic Tac Go board from a screenshot using Gemini Vision.

    Retry strategy:
      1. First attempt: send the image with PROMPT. If the response validates,
         return the board immediately.
      2. If validation fails, make a second call using RETRY_PROMPT with the
         first (bad) response appended. Gemini can see what it produced and why
         it was wrong, which usually leads to a corrected output.

    If the retry also fails validation, the BoardParseError from the second call
    propagates to the caller.

    Raises FileNotFoundError if the image path doesn't exist.
    """
    image_path = str(image_path)
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Screenshot does not exist: {image_path}")

    first_response = _call_gemini(image_path, PROMPT, model)
    try:
        return board_from_response_text(first_response)
    except BoardParseError:
        # Feed the bad response back to Gemini so it can see what went wrong.
        retry_response = _call_gemini(
            image_path,
            RETRY_PROMPT + first_response,
            model,
        )
        return board_from_response_text(retry_response)


def print_board(board):
    """Pretty-print a board to stdout, showing "." for empty cells."""
    for row in board:
        print(" ".join(cell if cell else "." for cell in row))


def main():
    parser = argparse.ArgumentParser(description="Parse a Google Tic Tac Go screenshot into a padded 8x8 board.")
    parser.add_argument("image_path", help="Path to a saved screenshot.")
    parser.add_argument(
        "--model",
        default="gemini-flash-latest",
        help="Gemini model to use. Defaults to gemini-flash-latest.",
    )
    args = parser.parse_args()

    board = parse_board_from_image(args.image_path, model=args.model)
    print(json.dumps({"board": board}, indent=2))
    print()
    print_board(board)


if __name__ == "__main__":
    main()
