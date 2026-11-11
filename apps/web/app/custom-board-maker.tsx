"use client";

import { useMemo, useState } from "react";
import { ArrowRight, Eraser, LoaderCircle, RotateCcw, Square } from "lucide-react";
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

const PIECES: { value: Cell; label: string }[] = [
  { value: "U", label: "Player" }, { value: "O", label: "O" },
  { value: "X", label: "X" }, { value: "B", label: "Wall" },
  { value: "", label: "Erase" },
];
const EDIT_HELP = "Choose a piece, then tap a square to place it.";

function PieceMark({ cell }: { cell: Cell }) {
  if (cell === "") return <Eraser size={18} aria-hidden="true" />;
  if (cell === "B") return <Square size={18} fill="currentColor" aria-hidden="true" />;
  return <span className={`custom-piece replay-piece-${cell.toLowerCase()}`} aria-hidden="true"><span className="piece-mark" /></span>;
}

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
  const [selectedPiece, setSelectedPiece] = useState<Cell>("X");
  const [message, setMessage] = useState(EDIT_HELP);
  const [loading, setLoading] = useState(false);
  const frames = useMemo(() => result ? buildReplayFrames(result.start_board, result.moves) : [], [result]);

  function resize(nextRows: number, nextColumns: number) {
    const safeRows = Math.max(3, Math.min(8, Number.isFinite(nextRows) ? Math.trunc(nextRows) : rows));
    const safeColumns = Math.max(3, Math.min(8, Number.isFinite(nextColumns) ? Math.trunc(nextColumns) : columns));
    const next = Array.from({ length: safeRows }, (_, row) =>
      Array.from({ length: safeColumns }, (_, column) => board[row]?.[column] ?? ""),
    );
    if (!next.flat().includes("U")) next[0]![0] = "U";
    setRows(safeRows);
    setColumns(safeColumns);
    setBoard(next);
    setResult(null);
    setMessage(EDIT_HELP);
  }

  function placePiece(row: number, column: number) {
    setBoard((current) => current.map((line, rowIndex) => line.map((cell, columnIndex) => {
      if (rowIndex === row && columnIndex === column) return selectedPiece;
      return selectedPiece === "U" && cell === "U" ? "" : cell;
    })));
    setResult(null);
    setMessage(EDIT_HELP);
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
      <h2 id="custom-board-title">Build your board</h2>
      <fieldset className="custom-editor" disabled={loading}>
        <legend className="sr-only">Board size and pieces</legend>
        <div className="custom-size-controls">
          <label>Rows <select value={rows} onChange={(event) => resize(Number(event.target.value), columns)}>{[3, 4, 5, 6, 7, 8].map((size) => <option key={size}>{size}</option>)}</select></label>
          <label>Columns <select value={columns} onChange={(event) => resize(rows, Number(event.target.value))}>{[3, 4, 5, 6, 7, 8].map((size) => <option key={size}>{size}</option>)}</select></label>
          <button type="button" className="custom-reset" onClick={() => { setBoard(blankBoard(rows, columns)); setResult(null); setMessage(EDIT_HELP); }}><RotateCcw size={15} aria-hidden="true" /> Reset board</button>
        </div>
        <div className="custom-palette" role="group" aria-label="Choose a piece">
          {PIECES.map(({ value, label }) => <button key={label} type="button" className="custom-tool" aria-pressed={selectedPiece === value} onClick={() => setSelectedPiece(value)}><PieceMark cell={value} /><span>{label}</span></button>)}
        </div>
        <p className="custom-instructions" id="custom-instructions">Choose a piece. Tap a square. Make it yours.</p>
        <div className="custom-board" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }} aria-label="Editable Tic Tac Go board">
          {board.flatMap((line, row) => line.map((cell, column) => (
            <button key={`${row}-${column}`} type="button" className={`custom-cell custom-cell-${cell || "empty"}`} onClick={() => placePiece(row, column)} aria-label={`Row ${row + 1}, column ${column + 1}: ${cell ? PIECES.find((piece) => piece.value === cell)?.label : "empty"}`} aria-describedby="custom-instructions">
              {cell !== "" && <PieceMark cell={cell} />}
            </button>
          )))}
        </div>
      </fieldset>
      <button type="button" className="custom-solve-button" onClick={solve} disabled={loading}>
        {loading ? <><LoaderCircle className="custom-loading" size={18} aria-hidden="true" /> Finding a route…</> : <>Solve this board <ArrowRight size={18} aria-hidden="true" /></>}
      </button>
      <p className="custom-status" role="status">{message}</p>
      {result && (
        <div className="custom-result">
          <p><strong>{result.solved ? "Verified solution" : "No route found"}</strong> · {result.states_checked.toLocaleString()} states · {(result.elapsed_ms / 1000).toFixed(2)} s</p>
          {result.solved && <SolvePlayer key={result.moves} frames={frames} emptyMessage="Board pending" />}
        </div>
      )}
    </section>
  );
}
