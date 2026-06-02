export type Cell = "" | "X" | "O" | "U" | "B";
export type Direction = "U" | "D" | "L" | "R";
export type ReplayStatus = "ready" | "won" | "lost";
export type ReplayPieceKind = Exclude<Cell, "" | "B">;

export type ReplayPiece = {
  id: string;
  kind: ReplayPieceKind;
  row: number;
  col: number;
};

export type ReplayFrame = {
  move: Direction | null;
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

export function visibleBoard(board: Cell[][]): Cell[][] {
  let rows = board.length;
  let cols = board[0]?.length ?? 1;

  while (rows > 1 && board[rows - 1]?.every((cell) => cell === "B")) rows -= 1;
  while (cols > 1 && board.slice(0, rows).every((row) => row[cols - 1] === "B")) cols -= 1;

  return board.slice(0, rows).map((row) => row.slice(0, cols));
}

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

export function applyReplayMove(current: ReplayFrame, move: Direction): ReplayFrame {
  const board = cloneBoard(current.board);
  const pieces = current.pieces.map((piece) => ({ ...piece }));
  const [rowDelta, colDelta] = DIRECTIONS[move];
  const [userRow, userCol] = findUser(board);
  const nextRow = userRow + rowDelta;
  const nextCol = userCol + colDelta;
  const pushRow = userRow + rowDelta * 2;
  const pushCol = userCol + colDelta * 2;

  if (!isInBounds(board, nextRow, nextCol) || board[nextRow][nextCol] === "B") {
    return frame(move, board, pieces);
  }

  const destination = board[nextRow][nextCol];
  if (MOVABLE_PIECES.has(destination)) {
    if (!isInBounds(board, pushRow, pushCol) || board[pushRow][pushCol] !== "") {
      return frame(move, board, pieces);
    }

    const pushedPiece = pieces.find((piece) => piece.row === nextRow && piece.col === nextCol);
    if (!pushedPiece) throw new Error("Replay piece layer is out of sync with the board.");
    pushedPiece.row = pushRow;
    pushedPiece.col = pushCol;
    board[pushRow][pushCol] = destination;
  }

  const user = pieces.find((piece) => piece.kind === "U");
  if (!user) throw new Error("Replay piece layer must contain one user piece.");
  user.row = nextRow;
  user.col = nextCol;
  board[nextRow][nextCol] = "U";
  board[userRow][userCol] = "";

  return frame(move, board, pieces);
}

export function buildReplayFrames(board: Cell[][] | null, moves: string | null): ReplayFrame[] {
  if (!board) return [];

  const frames = [createReplayStart(board)];
  for (const move of moves ?? "") {
    if (!(move in DIRECTIONS)) throw new Error(`Unsupported replay move: ${move}`);
    frames.push(applyReplayMove(frames[frames.length - 1], move as Direction));
  }
  return frames;
}
