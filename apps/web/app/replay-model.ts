// Pure data model for stepping through a solve replay.
// No React or browser dependencies — safe to run on the server or in tests.
//
// A "replay" is a sequence of ReplayFrames derived from an initial board and a
// moves string (e.g. "LLUU"). Each frame holds the full board state plus a flat
// list of pieces with stable IDs, which the board renderer uses as React keys
// so CSS transitions animate smoothly instead of re-mounting elements.

// Cell values:
//   ""  — empty cell
//   "X" — enemy piece (pushable, counts toward a losing three-in-a-row)
//   "O" — enemy piece that also counts toward a winning three-in-a-row when adjacent to U
//   "U" — the user-controlled piece (there is exactly one per board)
//   "B" — barrier / wall cell (impassable, hidden in the UI)
export type Cell = "" | "X" | "O" | "U" | "B";
export type Direction = "U" | "D" | "L" | "R";
export type ReplayStatus = "ready" | "won" | "lost";
export type ReplayPieceKind = Exclude<Cell, "" | "B">;

// A piece with a stable `id` (e.g. "o-1", "x-3") so React can track it across
// frames and fire CSS transitions on position changes rather than replacing the
// DOM element entirely.
export type ReplayPiece = {
  id: string;
  kind: ReplayPieceKind;
  row: number;
  col: number;
};

export type ReplayFrame = {
  move: Direction | null; // null for the initial frame (before any moves)
  board: Cell[][];
  pieces: ReplayPiece[];
  status: ReplayStatus;
};

const DIRECTIONS: Record<Direction, readonly [number, number]> = {
  U: [-1, 0],
  D: [1, 0],
  L: [0, -1],
  R: [0, 1],
};

const MOVABLE_PIECES = new Set<Cell>(["X", "O"]);
const USEFUL_PIECES = new Set<Cell>(["O", "U"]);

function cloneBoard(board: Cell[][]): Cell[][] {
  return board.map((row) => [...row]);
}

function isInBounds(board: Cell[][], row: number, col: number): boolean {
  return row >= 0 && row < board.length && col >= 0 && col < (board[0]?.length ?? 0);
}

function findUser(board: Cell[][]): [number, number] {
  for (let row = 0; row < board.length; row += 1) {
    for (let col = 0; col < (board[row]?.length ?? 0); col += 1) {
      if (board[row][col] === "U") return [row, col];
    }
  }
  throw new Error("Replay board must contain one user piece.");
}

// Checks for any horizontal or vertical run of three consecutive cells that
// all satisfy `test`. Used to detect win/loss conditions.
function lineMatches(board: Cell[][], test: (cell: Cell) => boolean): boolean {
  const rows = board.length;
  const cols = board[0]?.length ?? 0;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col <= cols - 3; col += 1) {
      if (test(board[row][col]) && test(board[row][col + 1]) && test(board[row][col + 2])) {
        return true;
      }
    }
  }

  for (let row = 0; row <= rows - 3; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      if (test(board[row][col]) && test(board[row + 1][col]) && test(board[row + 2][col])) {
        return true;
      }
    }
  }

  return false;
}

export function replayStatus(board: Cell[][]): ReplayStatus {
  if (lineMatches(board, (cell) => USEFUL_PIECES.has(cell))) return "won";
  if (lineMatches(board, (cell) => cell === "X")) return "lost";
  return "ready";
}

// Boards are internally padded to 8×8, but trailing rows/columns that are
// entirely barriers are invisible in the game. This trims those padding rows
// and columns from the bottom-right so the UI only shows the live play area.
export function visibleBoard(board: Cell[][]): Cell[][] {
  let rows = board.length;
  let cols = board[0]?.length ?? 1;

  while (rows > 1 && board[rows - 1]?.every((cell) => cell === "B")) rows -= 1;
  while (cols > 1 && board.slice(0, rows).every((row) => row[cols - 1] === "B")) cols -= 1;

  return board.slice(0, rows).map((row) => row.slice(0, cols));
}

// Walks the board top-left to bottom-right and assigns each piece a stable ID
// like "x-1", "o-2", "u-1". The scan order must stay consistent across calls
// (it always is, since board layout doesn't change between frames) so that
// piece IDs don't shift and CSS transitions animate the right elements.
function hydratePieces(board: Cell[][]): ReplayPiece[] {
  const counts: Record<ReplayPieceKind, number> = { X: 0, O: 0, U: 0 };
  const pieces: ReplayPiece[] = [];

  board.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      if (cell === "" || cell === "B") return;
      counts[cell] += 1;
      pieces.push({
        id: `${cell.toLowerCase()}-${counts[cell]}`,
        kind: cell,
        row: rowIndex,
        col: colIndex,
      });
    });
  });

  return pieces;
}

function frame(move: Direction | null, board: Cell[][], pieces: ReplayPiece[]): ReplayFrame {
  return {
    move,
    board,
    pieces,
    status: replayStatus(board),
  };
}

export function createReplayStart(board: Cell[][]): ReplayFrame {
  const croppedBoard = visibleBoard(cloneBoard(board));
  return frame(null, croppedBoard, hydratePieces(croppedBoard));
}

// Applies one move to the current frame and returns the resulting frame.
//
// Push logic: if the cell the user is moving into contains a movable piece
// (X or O), we try to push that piece one further step in the same direction.
// If the landing cell for the pushed piece is out-of-bounds, a barrier, or
// already occupied, the entire move is blocked and the board is returned
// unchanged. Only the user piece (U) advances — it never pushes more than one
// piece per move.
export function applyReplayMove(current: ReplayFrame, move: Direction): ReplayFrame {
  const board = cloneBoard(current.board);
  const pieces = current.pieces.map((piece) => ({ ...piece }));
  const [rowDelta, colDelta] = DIRECTIONS[move];
  const [userRow, userCol] = findUser(board);
  const nextRow = userRow + rowDelta;
  const nextCol = userCol + colDelta;

  // Where the pushed piece would land (two steps from the user's current cell)
  const pushRow = userRow + rowDelta * 2;
  const pushCol = userCol + colDelta * 2;

  // Can't move into a barrier or off the board
  if (!isInBounds(board, nextRow, nextCol) || board[nextRow][nextCol] === "B") {
    return frame(move, board, pieces);
  }

  const destination = board[nextRow][nextCol];
  if (MOVABLE_PIECES.has(destination)) {
    // There's a piece in the way — check whether it can be pushed
    if (!isInBounds(board, pushRow, pushCol) || board[pushRow][pushCol] !== "") {
      return frame(move, board, pieces); // blocked
    }

    const pushedPiece = pieces.find((piece) => piece.row === nextRow && piece.col === nextCol);
    if (!pushedPiece) throw new Error("Replay piece layer is out of sync with the board.");
    pushedPiece.row = pushRow;
    pushedPiece.col = pushCol;
    board[pushRow][pushCol] = destination;
  }

  // Move the user piece into the now-vacated (or always-empty) next cell
  const user = pieces.find((piece) => piece.kind === "U");
  if (!user) throw new Error("Replay piece layer must contain one user piece.");
  user.row = nextRow;
  user.col = nextCol;
  board[nextRow][nextCol] = "U";
  board[userRow][userCol] = "";

  return frame(move, board, pieces);
}

// Converts a raw moves string (e.g. "LLUU") and an initial board into the full
// sequence of ReplayFrames that the player steps through. Frame 0 is always
// the starting position; frames 1..N each correspond to one move.
export function buildReplayFrames(board: Cell[][] | null, moves: string | null): ReplayFrame[] {
  if (!board) return [];

  const frames = [createReplayStart(board)];
  for (const move of moves ?? "") {
    if (!(move in DIRECTIONS)) throw new Error(`Unsupported replay move: ${move}`);
    frames.push(applyReplayMove(frames[frames.length - 1], move as Direction));
  }
  return frames;
}
