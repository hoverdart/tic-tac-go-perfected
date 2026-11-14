import type { SolutionRecord } from "./game-view";
import { getTodaySolution, getSolutionByDate } from "./solution-data";
import { getOptimizedSolutions, matchesOptimizedBoard, optimizedSolutionRecord, type OptimizedSolution } from "./optimized-solution-data";

// The verified feed is a quality overlay, not a replacement for board identity
// or daily availability. Never swap a date-only match or regress a shorter solve.
export function preferOptimizedSolution(solution: SolutionRecord, optimized: OptimizedSolution | null): SolutionRecord {
  if (!optimized || !matchesOptimizedBoard(solution, optimized) ||
      (solution.status === "solved" && solution.moves !== null && solution.moves.length <= optimized.moves.length)) {
    return solution;
  }
  return optimizedSolutionRecord(optimized);
}

async function preferred(solution: SolutionRecord): Promise<SolutionRecord> {
  const rows = await getOptimizedSolutions(); // Verified bundled fallback on feed outage.
  return preferOptimizedSolution(solution, rows.find((row) => row.puzzle_date === solution.puzzle_date) ?? null);
}

export async function getPreferredTodaySolution() {
  const result = await getTodaySolution();
  return result.isDemo ? result : { ...result, solution: await preferred(result.solution) };
}

export async function getPreferredSolutionByDate(date: string) {
  const solution = await getSolutionByDate(date);
  return solution === null ? null : preferred(solution);
}
