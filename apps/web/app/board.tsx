// Renders a single Tic-Tac-Go board frame.
//
// The board is drawn in two layers that sit inside the same grid container:
//   1. A CSS Grid of <span> tiles — one per cell — that form the background
//      grid lines and handle barrier visibility. These never move.
//   2. A single absolutely-positioned "piece-layer" span that contains one
//      <span> per piece. Each piece is translated to its cell via CSS custom
//      properties (--piece-row / --piece-col) so the browser can animate moves
//      with a CSS transform without touching the grid tiles at all.
//
// Keeping the piece positions out of the grid avoids repainting the entire grid
// on every frame step, and lets CSS transitions on `transform` run on the
// compositor thread for smooth animation.
import type { CSSProperties } from "react";
import type { ReplayFrame, ReplayPiece } from "./replay-model";

type BoardProps = {
  frame: ReplayFrame | null;
  emptyMessage?: string;
};

// Sets the CSS custom properties that drive the piece's translate() animation
// in globals.css. The stylesheet reads --piece-row and --piece-col to compute
// the transform offset relative to the top-left cell.
function pieceStyle(piece: ReplayPiece): CSSProperties {
  return {
    ["--piece-row" as string]: piece.row,
    ["--piece-col" as string]: piece.col,
  };
}

export function Board({ frame, emptyMessage = "Board pending" }: BoardProps) {
  const rows = frame?.board.length ?? 6;
  const cols = frame?.board[0]?.length ?? 6;

  // --board-rows and --board-cols let the CSS Grid column/row counts stay
  // data-driven — we never hardcode the board size in the stylesheet.
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
        {/* Layer 1: background grid tiles. Barrier cells ("B") get
            tile-barrier which sets visibility: hidden — the cell still
            occupies space so the grid dimensions stay correct. */}
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

        {/* Layer 2: animated pieces (or a placeholder message when no frame
            is available yet). Each piece keeps a stable `id` across frames
            so React doesn't unmount/remount it on every step, which is what
            allows the CSS transition to fire. */}
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
