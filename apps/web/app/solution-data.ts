import type { HistoryEntry, SolutionRecord } from "./game-view";
import { getBackendBaseUrl } from "./backend-url";
import {
  SOLUTION_HISTORY_CACHE_TAG,
  SOLUTION_TODAY_CACHE_TAG,
  solutionDateCacheTag,
} from "./solution-cache";

export const demoSolution: SolutionRecord = {
  puzzle_date: new Date().toISOString().slice(0, 10),
  source_url: "",
  parser_name: "gemini",
  solver_name: "bfs",
  board: [
    ["", "", "", "", "", ""],
    ["", "O", "O", "", "", ""],
    ["", "", "X", "", "", ""],
    ["", "", "", "", "", "U"],
    ["", "B", "B", "", "", ""],
    ["", "", "", "", "", ""],
  ],
  moves: "LLUU",
  final_board: null,
  step_boards: [],
  states_checked: 128,
  elapsed_ms: 7,
  status: "solved",
  error_message: null,
  puzzle_title: null,
};

function unavailableSolution(date: string, errorMessage: string): SolutionRecord {
  return {
    ...demoSolution,
    puzzle_date: date,
    board: null,
    moves: null,
    states_checked: null,
    elapsed_ms: null,
    status: "failed",
    error_message: errorMessage,
  };
}

export function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

export function isIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

export function formatLongDate(value: string): string {
  if (!isIsoDate(value)) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${value}T00:00:00Z`));
}

export function solutionMetaDescription(
  solution: SolutionRecord,
  isToday: boolean,
): string {
  const date = formatLongDate(solution.puzzle_date);
  const timing = isToday ? "today's" : `the ${date}`;
  if (solution.status === "solved" && solution.moves !== null) {
    return `See ${timing} Tic Tac Go solution, including the verified ${solution.moves.length}-move sequence and interactive step-by-step board replay.`;
  }
  return `Check the Tic Tac Go puzzle status and board replay for ${date}.`;
}

export async function getTodaySolution(): Promise<{
  solution: SolutionRecord;
  isDemo: boolean;
}> {
  const apiBaseUrl = getBackendBaseUrl();
  const today = todayIsoDate();
  if (!apiBaseUrl) {
    return process.env.NODE_ENV === "development"
      ? { solution: demoSolution, isDemo: true }
      : {
          solution: unavailableSolution(today, "Backend URL is not configured."),
          isDemo: false,
        };
  }

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/today`, {
      cache: "force-cache",
      next: { tags: [SOLUTION_TODAY_CACHE_TAG] },
    });
    if (!response.ok) {
      return {
        solution: unavailableSolution(today, `Backend returned ${response.status}.`),
        isDemo: false,
      };
    }
    return { solution: await response.json(), isDemo: false };
  } catch (error) {
    return {
      solution: unavailableSolution(
        today,
        error instanceof Error ? error.message : "Could not reach backend.",
      ),
      isDemo: false,
    };
  }
}

export async function getSolutionByDate(date: string): Promise<SolutionRecord | null> {
  const apiBaseUrl = getBackendBaseUrl();
  if (!apiBaseUrl) {
    return unavailableSolution(date, "Backend URL is not configured.");
  }

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/${date}`, {
      cache: "force-cache",
      next: { tags: [solutionDateCacheTag(date)] },
    });
    if (!response.ok) return null;
    const solution: SolutionRecord = await response.json();
    if (solution.status === "pending" && date !== todayIsoDate()) return null;
    return solution;
  } catch (error) {
    return unavailableSolution(
      date,
      error instanceof Error ? error.message : "Could not reach backend.",
    );
  }
}

async function getHistory(limit: number): Promise<HistoryEntry[]> {
  const apiBaseUrl = getBackendBaseUrl();
  if (!apiBaseUrl) return [];

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/recent?limit=${limit}`, {
      cache: "force-cache",
      next: { tags: [SOLUTION_HISTORY_CACHE_TAG] },
    });
    if (!response.ok) return [];
    return await response.json();
  } catch {
    return [];
  }
}

export function getFullHistory(): Promise<HistoryEntry[]> {
  return getHistory(365);
}

export function getSitemapHistory(): Promise<HistoryEntry[]> {
  return getHistory(10_000);
}
