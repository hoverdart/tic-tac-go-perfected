export type Cell = "" | "X" | "O" | "U" | "B";

function visibleBoard(board: Cell[][] | null): Cell[][] | null {
  if (!board) return null;

  let lastRow = board.length - 1;
  let lastCol = board[0]?.length ? board[0].length - 1 : 0;

  while (lastRow > 0 && board[lastRow]?.every((cell) => cell === "B")) {
    lastRow -= 1;
  }

  while (
    lastCol > 0 &&
    board.slice(0, lastRow + 1).every((row) => row[lastCol] === "B")
  ) {
    lastCol -= 1;
  }

  return board.slice(0, lastRow + 1).map((row) => row.slice(0, lastCol + 1));
}

export function Piece({ cell }: { cell: Cell }) {
  if (cell === "" || cell === "B") return null;
  return <span className={`piece piece-${cell.toLowerCase()}`} aria-label={cell} />;
}

export function Board({
  board,
  compact = false,
}: {
  board: Cell[][] | null;
  compact?: boolean;
}) {
  const displayBoard = visibleBoard(board);

  if (!displayBoard) {
    return (
      <div className="board-shell board-empty">
        <div className="empty-board-copy">Board pending</div>
      </div>
    );
  }

  const cols = displayBoard[0]?.length || 1;

  return (
    <div className={compact ? "board-shell board-shell-small" : "board-shell"}>
      <div
        className={compact ? "board-grid board-grid-small" : "board-grid"}
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        aria-label="Tic Tac Go board"
      >
        {displayBoard.flatMap((row, rowIndex) =>
          row.map((cell, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`tile tile-${cell || "empty"}`}
              aria-label={cell || "empty"}
            >
              <Piece cell={cell} />
            </div>
          )),
        )}
      </div>
    </div>
  );
}
