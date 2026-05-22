import argparse
import json
import mimetypes
import os
from pathlib import Path


STANDARD_BOARD_SIZE = 8
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

RETRY_PROMPT = """The previous response did not validate as a Tic Tac Go board.

Return only corrected JSON for the visible board:
- a rectangular board from 3x3 through 8x8
- allowed cells only: X, O, U, B, .
- exactly one U

Read top-to-bottom, left-to-right. Do not solve the board.
Previous invalid response:
"""

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
    """Pad smaller boards with blocked cells so DQN observations stay 8x8."""
    padded = []
    for row in board:
        padded.append(row + ["B"] * (STANDARD_BOARD_SIZE - len(row)))

    while len(padded) < STANDARD_BOARD_SIZE:
        padded.append(["B"] * STANDARD_BOARD_SIZE)

    return padded


def validate_board(board):
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
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/png"


def _api_key_from_env_file():
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
    genai, types = _load_genai()
    image_bytes = Path(image_path).read_bytes()

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
    image_path = str(image_path)
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Screenshot does not exist: {image_path}")

    first_response = _call_gemini(image_path, PROMPT, model)
    try:
        return board_from_response_text(first_response)
    except BoardParseError:
        retry_response = _call_gemini(
            image_path,
            RETRY_PROMPT + first_response,
            model,
        )
        return board_from_response_text(retry_response)


def print_board(board):
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
