"use client";

import { useMemo, useState } from "react";
import { SolvePlayer } from "./solve-player";
import { buildReplayFrames, type Cell } from "./replay-model";

type CustomResult = {
  solved: boolean;
  solver_name: string;
  moves: string | null;
  states_checked: number;
  elapsed_ms: number;
  start_board: Cell[][];
  final_board: Cell[][] | null;
  steps: { move: string; board: Cell[][] }[];
  cached: boolean;
  remaining: number | null;
};

const CYCLE: Cell[] = ["", "X", "O", "U", "B"];

function blankBoard(rows: number, columns: number): Cell[][] {
  const board = Array.from({ length: rows }, () => Array<Cell>(columns).fill(""));
  board[0]![0] = "U";
  board[0]![1] = "O";
  board[0]![2] = "O";
  return board;
}

export function CustomBoardMaker() {
  const [rows, setRows] = useState(6);
  const [columns, setColumns] = useState(6);
  const [board, setBoard] = useState<Cell[][]>(() => blankBoard(6, 6));
  const [result, setResult] = useState<CustomResult | null>(null);
  const [message, setMessage] = useState("Click a tile to cycle: empty, X, O, player, wall.");
  const [loading, setLoading] = useState(false);
  const frames = useMemo(() => result ? buildReplayFrames(result.start_board, result.moves) : [], [result]);

  function resize(nextRows: number, nextColumns: number) {
    const safeRows = Math.max(3, Math.min(8, Number.isFinite(nextRows) ? nextRows : rows));
    const safeColumns = Math.max(3, Math.min(8, Number.isFinite(nextColumns) ? nextColumns : columns));
    const next = Array.from({ length: safeRows }, (_, row) =>
      Array.from({ length: safeColumns }, (_, column) => board[row]?.[column] ?? ""),
    );
    if (!next.flat().includes("U")) next[0]![0] = "U";
    setRows(safeRows);
    setColumns(safeColumns);
    setBoard(next);
    setResult(null);
  }

  function cycleCell(row: number, column: number) {
    setBoard((current) => current.map((line, rowIndex) => line.map((cell, columnIndex) => {
      if (rowIndex !== row || columnIndex !== column) return cell;
      const next = CYCLE[(CYCLE.indexOf(cell) + 1) % CYCLE.length]!;
      return next;
    })).map((line, rowIndex) => line.map((cell, columnIndex) =>
      cell === "U" && (rowIndex !== row || columnIndex !== column) && CYCLE[(CYCLE.indexOf(current[row]?.[column] ?? "") + 1) % CYCLE.length] === "U" ? "" : cell,
    )));
    setResult(null);
  }

  async function solve() {
    const playerCount = board.flat().filter((cell) => cell === "U").length;
    if (playerCount !== 1) {
      setMessage("Place exactly one player piece before solving.");
      return;
    }
    setLoading(true);
    setMessage("Checking this board with the verified push solver…");
    setResult(null);
    try {
      const response = await fetch("/api/custom-solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board }),
      });
      const payload = await response.json() as CustomResult | { detail?: string; error?: string };
      if (!response.ok) {
        const failure = payload as { detail?: string; error?: string };
        setMessage(failure.detail ?? failure.error ?? "The solver could not accept this board.");
        return;
      }
      const solved = payload as CustomResult;
      setResult(solved);
      setMessage(solved.solved
        ? `${solved.cached ? "Cached verified" : "Verified"} solution ready${solved.remaining === null ? "" : ` · ${solved.remaining} new solves left today`}.`
        : `No route was found within the five-second public search budget${solved.remaining === null ? "" : ` · ${solved.remaining} new solves left today`}.`);
    } catch {
      setMessage("Could not reach the solver. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="custom-solver" aria-labelledby="custom-board-title">
      <div className="custom-size-controls">
        <label>Rows <input type="number" min="3" max="8" value={rows} onChange={(event) => resize(Number(event.target.value), columns)} /></label>
        <label>Columns <input type="number" min="3" max="8" value={columns} onChange={(event) => resize(rows, Number(event.target.value))} /></label>
        <button type="button" className="hint-button" onClick={() => { setBoard(blankBoard(rows, columns)); setResult(null); }}>Reset board</button>
      </div>
      <div className="custom-board" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }} aria-label="Editable Tic Tac Go board">
        {board.flatMap((line, row) => line.map((cell, column) => (
          <button key={`${row}-${column}`} type="button" className={`custom-cell custom-cell-${cell || "empty"}`} onClick={() => cycleCell(row, column)} aria-label={`Row ${row + 1}, column ${column + 1}: ${cell || "empty"}`}>
            {cell === "U" ? "◉" : cell === "O" ? "○" : cell === "X" ? "×" : cell === "B" ? "■" : ""}
          </button>
        )))}
      </div>
      <button type="button" className="custom-solve-button" onClick={solve} disabled={loading}>
        {loading ? "Solving…" : "Solve this board"}
      </button>
      <p className="custom-status" role="status">{message}</p>
      {result && (
        <div className="custom-result">
          <p><strong>{result.solved ? "Verified solution" : "No route found"}</strong> · {result.states_checked.toLocaleString()} states · {(result.elapsed_ms / 1000).toFixed(2)} s</p>
          {result.solved && <SolvePlayer frames={frames} emptyMessage="Board pending" />}
        </div>
      )}
    </section>
  );
}
