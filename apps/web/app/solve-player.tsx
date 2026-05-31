"use client";

import { useEffect, useMemo } from "react";
import type { Cell } from "./board";

export type SolveFrame = {
  move: string;
  board: Cell[][];
};

function cropBoard(board: Cell[][], rows: number, cols: number) {
  return board.slice(0, rows).map((row) => row.slice(0, cols));
}

function visibleSize(board: Cell[][]) {
  let rows = board.length;
  let cols = board[0]?.length || 1;

  while (rows > 1 && board[rows - 1]?.every((cell) => cell === "B")) rows -= 1;
  while (cols > 1 && board.slice(0, rows).every((row) => row[cols - 1] === "B")) {
    cols -= 1;
  }

  return { rows, cols };
}

function userPosition(board: Cell[][]) {
  for (let row = 0; row < board.length; row += 1) {
    for (let col = 0; col < (board[row]?.length || 0); col += 1) {
      if (board[row][col] === "U") return { row, col };
    }
  }
  return { row: 0, col: 0 };
}

function Piece({ cell }: { cell: Cell }) {
  if (cell === "" || cell === "B" || cell === "U") return null;
  return <span className={`piece piece-${cell.toLowerCase()}`} aria-label={cell} />;
}

type SolvePlayerProps = {
  frames: SolveFrame[];
  index: number;
  playing: boolean;
  onSetIndex: (i: number) => void;
  onSetPlaying: (p: boolean) => void;
  showComplete: boolean;
};

export function SolvePlayer({
  frames,
  index,
  playing,
  onSetIndex,
  onSetPlaying,
  showComplete,
}: SolvePlayerProps) {
  const size = useMemo(
    () => visibleSize(frames[0]?.board || [[""]]),
    [frames],
  );
  const frame = frames[index] || frames[0];
  const board = frame ? cropBoard(frame.board, size.rows, size.cols) : [["" as Cell]];
  const user = userPosition(board);
  const hasReplay = frames.length > 1;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === " ") {
        e.preventDefault();
        onSetPlaying(!playing);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        onSetPlaying(false);
        onSetIndex(Math.max(index - 1, 0));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onSetPlaying(false);
        onSetIndex(Math.min(index + 1, frames.length - 1));
      } else if (e.key === "r" || e.key === "R") {
        onSetIndex(0);
        onSetPlaying(hasReplay);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playing, index, frames.length, hasReplay, onSetPlaying, onSetIndex]);

  function replay() {
    onSetIndex(0);
    onSetPlaying(hasReplay);
  }

  function step(delta: number) {
    onSetPlaying(false);
    onSetIndex(Math.min(Math.max(index + delta, 0), frames.length - 1));
  }

  return (
    <div className="solve-player">
      <div className="solve-player-topline">
        <span>{index === 0 ? "Start" : `Move ${index}`}</span>
        <strong>{frame?.move || "Ready"}</strong>
      </div>

      <div className="board-shell solve-board-shell">
        <div
          className="board-grid solve-board-grid"
          style={{
            gridTemplateColumns: `repeat(${size.cols}, minmax(0, 1fr))`,
            ["--rows" as string]: size.rows,
            ["--cols" as string]: size.cols,
            ["--user-row" as string]: user.row,
            ["--user-col" as string]: user.col,
          }}
          aria-label="Animated Tic Tac Go solution board"
        >
          {board.flatMap((row, rowIndex) =>
            row.map((cell, colIndex) => (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`tile tile-${cell === "U" ? "empty" : cell || "empty"}`}
                aria-label={cell || "empty"}
              >
                <Piece cell={cell} />
              </div>
            )),
          )}
          <span className="solve-user-piece piece piece-u" aria-label="U" />
        </div>
      </div>

      {showComplete && (
        <div className="solve-complete-badge" aria-live="polite">Solved ✓</div>
      )}

      <div className="solve-controls">
        <button
          type="button"
          className="solve-icon-btn"
          onClick={() => step(-1)}
          disabled={index === 0}
          aria-label="Step back"
        >
          <svg viewBox="0 0 16 16" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="10,4 6,8 10,12" />
          </svg>
        </button>

        <button
          type="button"
          className="solve-primary"
          onClick={() => onSetPlaying(!playing)}
          disabled={!hasReplay}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16">
              <rect x="3" y="3" width="4" height="10" rx="1" />
              <rect x="9" y="3" width="4" height="10" rx="1" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16">
              <polygon points="4,2 14,8 4,14" />
            </svg>
          )}
        </button>

        <button
          type="button"
          className="solve-icon-btn"
          onClick={() => step(1)}
          disabled={index >= frames.length - 1}
          aria-label="Step forward"
        >
          <svg viewBox="0 0 16 16" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6,4 10,8 6,12" />
          </svg>
        </button>

        <button
          type="button"
          className="solve-icon-btn"
          onClick={replay}
          disabled={!hasReplay}
          aria-label="Replay from start"
        >
          <svg viewBox="0 0 16 16" stroke="currentColor" fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2.5 8a5.5 5.5 0 1 1 1.1 3.3" />
            <polyline points="2.5,4.5 2.5,8 6,8" />
          </svg>
        </button>
      </div>

      <div className="solve-progress" aria-hidden="true">
        <span style={{ width: `${hasReplay ? (index / (frames.length - 1)) * 100 : 0}%` }} />
      </div>

      <p className="solve-keyboard-hint">Space · ← → · R</p>
    </div>
  );
}
