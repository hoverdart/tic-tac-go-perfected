import argparse
import json
from pathlib import Path


BOARD_SIZE = 6
ALLOWED_CELLS = {"X", "O", "U", "B", ""}


class OpenCVBoardParseError(ValueError):
    """Raised when OpenCV cannot confidently parse the board."""


def _load_cv2():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install OpenCV with "
            "`python3 -m pip install opencv-python`."
        ) from exc

    return cv2, np


def validate_board(board):
    if not isinstance(board, list) or len(board) != BOARD_SIZE:
        raise OpenCVBoardParseError(f"Board must have {BOARD_SIZE} rows.")

    user_count = 0
    normalized = []
    for row_index, row in enumerate(board):
        if not isinstance(row, list) or len(row) != BOARD_SIZE:
            raise OpenCVBoardParseError(
                f"Row {row_index} must have {BOARD_SIZE} cells."
            )

        normalized_row = []
        for col_index, cell in enumerate(row):
            if cell is None:
                cell = ""
            if not isinstance(cell, str):
                raise OpenCVBoardParseError(
                    f"Cell ({row_index}, {col_index}) must be a string."
                )

            cell = cell.strip().upper()
            if cell == ".":
                cell = ""
            if cell not in ALLOWED_CELLS:
                raise OpenCVBoardParseError(
                    f"Cell ({row_index}, {col_index}) has invalid value {cell!r}."
                )
            if cell == "U":
                user_count += 1
            normalized_row.append(cell)

        normalized.append(normalized_row)

    if user_count != 1:
        raise OpenCVBoardParseError(
            f"Board must contain exactly one U, got {user_count}."
        )

    return normalized


def _order_points(points, np):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp_board(image, contour, cv2, np):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    if len(approx) == 4:
        points = approx.reshape(4, 2).astype("float32")
    else:
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect).astype("float32")

    rect = _order_points(points, np)
    top_width = np.linalg.norm(rect[1] - rect[0])
    bottom_width = np.linalg.norm(rect[2] - rect[3])
    left_height = np.linalg.norm(rect[3] - rect[0])
    right_height = np.linalg.norm(rect[2] - rect[1])
    side = int(max(top_width, bottom_width, left_height, right_height))
    side = max(side, 300)

    destination = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (side, side))


def find_board_image(image):
    cv2, np = _load_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.03:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width == 0 or height == 0:
            continue

        aspect_ratio = width / height
        if 0.75 <= aspect_ratio <= 1.25:
            candidates.append((area, contour))

    if not candidates:
        raise OpenCVBoardParseError("Could not find a square Tic Tac Go board.")

    _, contour = max(candidates, key=lambda item: item[0])
    board = _warp_board(image, contour, cv2, np)
    return cv2.resize(board, (720, 720), interpolation=cv2.INTER_CUBIC)


def _foreground_mask(cell, cv2, np):
    inset_y = max(2, int(cell.shape[0] * 0.12))
    inset_x = max(2, int(cell.shape[1] * 0.12))
    inner = cell[inset_y:-inset_y, inset_x:-inset_x]

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]

    dark_mask = gray < 145
    saturated_mask = saturation > 45
    mask = np.where(dark_mask | saturated_mask, 255, 0).astype("uint8")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return inner, mask


def _diagonal_scores(mask, np):
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0

    positive_band = np.abs(ys - (xs * height / width))
    negative_band = np.abs(ys - ((width - xs) * height / width))
    tolerance = max(height, width) * 0.13
    positive_score = float(np.mean(positive_band < tolerance))
    negative_score = float(np.mean(negative_band < tolerance))
    return positive_score, negative_score


def _hole_score(mask, cv2, np):
    height, width = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    center = holes[
        int(height * 0.25): int(height * 0.75),
        int(width * 0.25): int(width * 0.75),
    ]
    return float(np.count_nonzero(center) / center.size)


def classify_cell(cell):
    cv2, np = _load_cv2()
    inner, mask = _foreground_mask(cell, cv2, np)
    foreground_ratio = float(np.count_nonzero(mask) / mask.size)
    if foreground_ratio < 0.025:
        return ""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    x, y, width, height = cv2.boundingRect(largest)
    bbox_area = max(1, width * height)
    extent = area / bbox_area
    cell_span = max(inner.shape[:2])
    bbox_span = max(width, height) / cell_span

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    foreground_pixels = hsv[mask > 0]
    mean_hue = float(np.mean(foreground_pixels[:, 0])) if len(foreground_pixels) else 0
    mean_saturation = (
        float(np.mean(foreground_pixels[:, 1])) if len(foreground_pixels) else 0
    )

    if foreground_ratio > 0.35 and extent > 0.55:
        return "B"

    # Google often renders the movable user piece with a more colorful fill
    # than X/O pieces. Keep this rule before shape rules.
    if mean_saturation > 85 and 75 <= mean_hue <= 145 and foreground_ratio > 0.08:
        return "U"

    positive_diag, negative_diag = _diagonal_scores(mask, np)
    if positive_diag > 0.32 and negative_diag > 0.32:
        return "X"

    hole_score = _hole_score(mask, cv2, np)
    perimeter = cv2.arcLength(largest, True)
    circularity = 0.0
    if perimeter:
        circularity = float(4 * np.pi * area / (perimeter * perimeter))
    if hole_score > 0.08 or (circularity > 0.45 and bbox_span > 0.35):
        return "O"

    if foreground_ratio > 0.12 and extent > 0.32:
        return "B"

    raise OpenCVBoardParseError("Found a piece but could not classify it.")


def parse_board_with_opencv(image_path, debug_dir=None):
    cv2, _ = _load_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read screenshot: {image_path}")

    board_image = find_board_image(image)
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path / "detected_board.png"), board_image)

    cell_size = board_image.shape[0] // BOARD_SIZE
    board = []
    for row in range(BOARD_SIZE):
        board_row = []
        for col in range(BOARD_SIZE):
            y1 = row * cell_size
            x1 = col * cell_size
            cell = board_image[y1:y1 + cell_size, x1:x1 + cell_size]
            board_row.append(classify_cell(cell))
        board.append(board_row)

    return validate_board(board)


def parse_board_from_image(image_path, use_fallback=True, debug_dir=None):
    try:
        return parse_board_with_opencv(image_path, debug_dir=debug_dir)
    except Exception as exc:
        if not use_fallback:
            raise

        try:
            from fallbackBoardParser import parse_board_from_image as fallback_parse
        except ImportError:
            raise exc

        return fallback_parse(image_path)


def print_board(board):
    for row in board:
        print(" ".join(cell if cell else "." for cell in row))


def main():
    parser = argparse.ArgumentParser(
        description="Parse a Google Tic Tac Go screenshot with OpenCV."
    )
    parser.add_argument("image_path", help="Path to a saved screenshot.")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not call fallbackBoardParser if OpenCV parsing fails.",
    )
    parser.add_argument(
        "--debug-dir",
        help="Optional directory where the detected board crop is written.",
    )
    args = parser.parse_args()

    board = parse_board_from_image(
        args.image_path,
        use_fallback=not args.no_fallback,
        debug_dir=args.debug_dir,
    )
    print(json.dumps({"board": board}, indent=2))
    print()
    print_board(board)


if __name__ == "__main__":
    main()
