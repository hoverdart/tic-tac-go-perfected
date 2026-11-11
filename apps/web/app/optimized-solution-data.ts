import { cache } from "react";
import fallbackFeed from "../data/optimized-v1.json";
import { buildReplayFrames, type Cell } from "./replay-model";
import type { SolutionRecord } from "./game-view";
import { isIsoDate, todayIsoDate } from "./solution-data";

export type OptimizedSolution = {
  puzzle_date: string;
  puzzle_title: string;
  board_sha256: string;
  board: Cell[][];
  moves: string;
  solver_name: "optimized-v1";
  optimality_proven: boolean;
  lower_bound: number;
  updated_at: string;
};

export const OPTIMIZED_FEED_URL = "https://raw.githubusercontent.com/hoverdart/tic-tac-go-perfected/optimized-v1-data/optimized-v1.json";

function hasXLine(board: Cell[][]): boolean {
  return board.some((row, r) => row.some((_, c) =>
    (row[c] === "X" && row[c + 1] === "X" && row[c + 2] === "X") ||
    (row[c] === "X" && board[r + 1]?.[c] === "X" && board[r + 2]?.[c] === "X"),
  ));
}

export function parseOptimizedFeed(value: unknown): OptimizedSolution[] {
  if (!value || typeof value !== "object") throw new Error("Invalid optimized feed.");
  const feed = value as Record<string, unknown>;
  if (feed.schema !== "optimized-v1-feed-v1" || feed.solver_name !== "optimized-v1" || !Array.isArray(feed.solutions)) {
    throw new Error("Unsupported optimized feed.");
  }
  const dates = new Set<string>();
  return feed.solutions.map((value: unknown) => {
    if (!value || typeof value !== "object") throw new Error("Invalid optimized solution.");
    const row = value as OptimizedSolution;
    if (typeof row.puzzle_date !== "string" || !isIsoDate(row.puzzle_date) || dates.has(row.puzzle_date) ||
        typeof row.puzzle_title !== "string" || !row.puzzle_title.trim() || row.puzzle_title.length > 80 ||
        row.solver_name !== "optimized-v1" || typeof row.board_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(row.board_sha256) ||
        !Array.isArray(row.board) || row.board.length !== 8 || row.board.some((line) => !Array.isArray(line) || line.length !== 8 || line.some((cell) => !["", "U", "O", "X", "B"].includes(cell))) ||
        row.board.flat().filter((cell) => cell === "U").length !== 1 ||
        typeof row.moves !== "string" || row.moves.length > 10000 || !/^[UDLR]*$/.test(row.moves) ||
        typeof row.optimality_proven !== "boolean" || !Number.isInteger(row.lower_bound) || row.lower_bound < 0 || row.lower_bound > row.moves.length ||
        (row.optimality_proven && row.lower_bound !== row.moves.length) ||
        typeof row.updated_at !== "string" || !Number.isFinite(Date.parse(row.updated_at))) {
      throw new Error("Invalid optimized solution fields.");
    }
    const frames = buildReplayFrames(row.board, row.moves);
    if (frames.at(-1)?.status !== "won" || frames.some((frame, i) =>
      hasXLine(frame.board) || (i < frames.length - 1 && frame.status === "won") ||
      (i > 0 && JSON.stringify(frame.board) === JSON.stringify(frames[i - 1].board)),
    )) throw new Error("Optimized solution failed replay verification.");
    dates.add(row.puzzle_date);
    return row;
  });
}

export function mergeOptimizedSolutions(base: OptimizedSolution[], incoming: OptimizedSolution[]): OptimizedSolution[] {
  const fallback = new Map(base.map((row) => [row.puzzle_date, row]));
  const rows = new Map<string, OptimizedSolution>();
  for (const row of incoming) {
    const old = fallback.get(row.puzzle_date);
    if (old && old.board_sha256 === row.board_sha256) {
      if (JSON.stringify(old.board) !== JSON.stringify(row.board) ||
          (old.optimality_proven && row.moves.length < old.moves.length) ||
          (row.optimality_proven && old.moves.length < row.moves.length)) {
        throw new Error("Optimized feed contradicts its bundled certificate or board.");
      }
      if (old.moves.length < row.moves.length ||
          (old.moves.length === row.moves.length && old.optimality_proven && !row.optimality_proven)) {
        rows.set(row.puzzle_date, old);
        continue;
      }
    }
    rows.set(row.puzzle_date, row);
  }
  return [...rows.values()].filter((row) => row.puzzle_date <= todayIsoDate())
    .sort((a, b) => b.puzzle_date.localeCompare(a.puzzle_date));
}

const loadFeed = cache(async () => {
  const base = parseOptimizedFeed(fallbackFeed);
  const url = process.env.OPTIMIZED_V1_FEED_URL ?? OPTIMIZED_FEED_URL;
  if (!url) return { solutions: mergeOptimizedSolutions([], base), error: null };
  try {
    const response = await fetch(url, { cache: "force-cache", next: { revalidate: 300, tags: ["optimized-v1"] }, signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(`Optimized feed returned ${response.status}.`);
    return { solutions: mergeOptimizedSolutions(base, parseOptimizedFeed(await response.json())), error: null };
  } catch (error) {
    return { solutions: mergeOptimizedSolutions([], base), error };
  }
});

export async function getOptimizedSolutions(required = false): Promise<OptimizedSolution[]> {
  const feed = await loadFeed();
  if (required && feed.error) throw feed.error;
  return feed.solutions;
}

export async function getOptimizedSolution(date: string): Promise<OptimizedSolution | null> {
  if (!isIsoDate(date) || date > todayIsoDate()) return null;
  const feed = await loadFeed();
  const row = feed.solutions.find((row) => row.puzzle_date === date);
  if (row) return row;
  // Preserve an ISR page for newer feed-only dates during an upstream outage.
  if (feed.error) throw feed.error;
  return null;
}

export function matchesOptimizedBoard(solution: SolutionRecord, optimized: OptimizedSolution | null): boolean {
  return optimized !== null && optimized.puzzle_date === solution.puzzle_date &&
    JSON.stringify(solution.board) === JSON.stringify(optimized.board);
}

export function optimizedSolutionRecord(row: OptimizedSolution): SolutionRecord {
  const frames = buildReplayFrames(row.board, row.moves);
  return {
    puzzle_date: row.puzzle_date, puzzle_title: row.puzzle_title,
    source_url: "https://dailygames.discover.google.com/games/tic-tac-go",
    parser_name: "catalog", solver_name: "optimized-v1", board: row.board, moves: row.moves,
    final_board: frames.at(-1)!.board,
    step_boards: frames.slice(1).map((frame) => ({ move: frame.move!, board: frame.board })),
    states_checked: null, elapsed_ms: null, status: "solved", error_message: null,
  };
}
