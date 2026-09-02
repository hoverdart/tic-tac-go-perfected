import type { Cell } from "./replay-model";

type Step = { move: string; board: Cell[][] };

export type ExplanationPhase = {
  moveIndex: number;
  title: string;
  detail: string;
};

export type SolutionExplanation = {
  goal: string | null;
  phases: ExplanationPhase[];
};

const DIRECTION: Record<string, string> = { U: "up", D: "down", L: "left", R: "right" };

function coordinate(row: number, column: number): string {
  return `${String.fromCharCode(65 + column)}${row + 1}`;
}

function findWinningLine(board: Cell[][] | null): string | null {
  if (!board) return null;
  const isO = (cell: Cell | undefined) => cell === "O" || cell === "U";
  for (let row = 0; row < board.length; row += 1) {
    for (let column = 0; column <= (board[row]?.length ?? 0) - 3; column += 1) {
      if (isO(board[row]?.[column]) && isO(board[row]?.[column + 1]) && isO(board[row]?.[column + 2])) {
        return `Finish the horizontal O line from ${coordinate(row, column)} to ${coordinate(row, column + 2)}.`;
      }
    }
  }
  const width = Math.max(0, ...board.map((row) => row.length));
  for (let column = 0; column < width; column += 1) {
    for (let row = 0; row <= board.length - 3; row += 1) {
      if (isO(board[row]?.[column]) && isO(board[row + 1]?.[column]) && isO(board[row + 2]?.[column])) {
        return `Finish the vertical O line from ${coordinate(row, column)} to ${coordinate(row + 2, column)}.`;
      }
    }
  }
  return null;
}

function movedPiece(before: Cell[][], after: Cell[][]): { piece: "O" | "X"; from: string; to: string } | null {
  // A pushed piece is replaced by U, so it does not appear as a conventional
  // O/X departure. Recover that move directly from the replay state instead.
  const rows = Math.max(before.length, after.length);
  const columns = Math.max(0, ...before.map((row) => row.length), ...after.map((row) => row.length));
  let pushedFrom: { row: number; column: number; piece: "O" | "X" } | null = null;
  let pushedTo: { row: number; column: number; piece: "O" | "X" } | null = null;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const previous = before[row]?.[column];
      const next = after[row]?.[column];
      if (next === "U" && (previous === "O" || previous === "X")) {
        pushedFrom = { row, column, piece: previous };
      }
      if (previous === "" && (next === "O" || next === "X")) {
        pushedTo = { row, column, piece: next };
      }
    }
  }
  if (pushedFrom && pushedTo && pushedFrom.piece === pushedTo.piece) {
    return {
      piece: pushedFrom.piece,
      from: coordinate(pushedFrom.row, pushedFrom.column),
      to: coordinate(pushedTo.row, pushedTo.column),
    };
  }

  const changes: { row: number; column: number; before: Cell | undefined; after: Cell | undefined }[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const previous = before[row]?.[column];
      const next = after[row]?.[column];
      if (previous !== next && previous !== "U" && next !== "U") {
        changes.push({ row, column, before: previous, after: next });
      }
    }
  }
  const from = changes.find((change) => change.before === "O" || change.before === "X");
  const to = changes.find((change) => change.after === "O" || change.after === "X");
  if (
    !from
    || !to
    || (from.before !== "O" && from.before !== "X")
    || from.before !== to.after
  ) return null;
  return { piece: from.before, from: coordinate(from.row, from.column), to: coordinate(to.row, to.column) };
}

export function buildSolutionExplanation(
  board: Cell[][] | null,
  finalBoard: Cell[][] | null,
  steps: Step[],
): SolutionExplanation {
  const phases: ExplanationPhase[] = [];
  let previous = board;
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index]!;
    const moved = previous ? movedPiece(previous, step.board) : null;
    if (moved) {
      phases.push({
        moveIndex: index + 1,
        title: `Move ${index + 1}: push the ${moved.piece}`,
        detail: `Press ${DIRECTION[step.move] ?? step.move} to move the ${moved.piece} from ${moved.from} to ${moved.to}.`,
      });
    }
    previous = step.board;
  }
  return { goal: findWinningLine(finalBoard), phases };
}
