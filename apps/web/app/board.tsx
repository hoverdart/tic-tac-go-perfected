import type { CSSProperties } from "react";
import type { ReplayFrame, ReplayPiece } from "./replay-model";

type BoardProps = {
  frame: ReplayFrame | null;
  emptyMessage?: string;
};

function pieceStyle(piece: ReplayPiece): CSSProperties {
  return {
    ["--piece-row" as string]: piece.row,
    ["--piece-col" as string]: piece.col,
  };
}

export function Board({ frame, emptyMessage = "Board pending" }: BoardProps) {
  const rows = frame?.board.length ?? 6;
  const cols = frame?.board[0]?.length ?? 6;
  const gridStyle = {
    ["--board-rows" as string]: rows,
    ["--board-cols" as string]: cols,
  };

  return (
    <div className="board-shell">
      <div
        className={`board-grid${frame ? "" : " board-grid-empty"}`}
        style={gridStyle}
        aria-label={frame ? "Animated Tic-Tac-Go solution board" : emptyMessage}
      >
        {(frame?.board ?? Array.from({ length: rows }, () => Array(cols).fill(""))).flatMap(
          (row, rowIndex) =>
            row.map((cell, colIndex) => (
              <span
                key={`${rowIndex}-${colIndex}`}
                className={`tile${cell === "B" ? " tile-barrier" : ""}`}
                aria-hidden="true"
              />
            )),
        )}

        {frame ? (
          <span className="piece-layer" aria-hidden="true">
            {frame.pieces.map((piece) => (
              <span
                key={piece.id}
                className={`replay-piece replay-piece-${piece.kind.toLowerCase()}`}
                style={pieceStyle(piece)}
              >
                <span className="piece-mark" />
              </span>
            ))}
          </span>
        ) : (
          <span className="board-empty-copy">{emptyMessage}</span>
        )}
      </div>
    </div>
  );
}
