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
  const title = solution.puzzle_title ? ` for ${solution.puzzle_title}` : "";
  if (solution.status === "solved" && solution.moves !== null) {
    return `See ${timing} Tic Tac Go solution${title}, including verified hint-first steps and the ${solution.moves.length}-move replay.`;
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
  if (!isIsoDate(date)) return null;
  const apiBaseUrl = getBackendBaseUrl();
  if (!apiBaseUrl) {
    throw new Error("Solution service is not configured.");
  }

  // Only a genuinely missing puzzle may become a cached 404/noindex page.
  // Throw on outages so ISR retains a previous good page and retries later.
  const response = await fetch(`${apiBaseUrl}/solutions/${date}`, {
    cache: "force-cache",
    next: { tags: [solutionDateCacheTag(date)], revalidate: 3600 },
  });
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`Solution service returned ${response.status}.`);
  }
  const solution: SolutionRecord = await response.json();
  if (solution.puzzle_date !== date) throw new Error("Solution date mismatch.");
  if (solution.status === "pending" && date !== todayIsoDate()) return null;
  return solution;
}

async function getHistory(limit: number, required = false): Promise<HistoryEntry[]> {
  const apiBaseUrl = getBackendBaseUrl();
  if (!apiBaseUrl) {
    if (required) throw new Error("Solution service is not configured.");
    return [];
  }

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/recent?limit=${limit}`, {
      cache: "force-cache",
      next: { tags: [SOLUTION_HISTORY_CACHE_TAG], revalidate: 3600 },
    });
    if (!response.ok) throw new Error(`Solution history returned ${response.status}.`);
    return await response.json();
  } catch (error) {
    if (required) throw error;
    return [];
  }
}

export function getFullHistory(): Promise<HistoryEntry[]> {
  return getHistory(10_000);
}

export function getSitemapHistory(): Promise<HistoryEntry[]> {
  // Never publish an almost-empty sitemap during a temporary backend outage.
  return getHistory(10_000, true);
}

export function nearbyHistory(history: HistoryEntry[], date: string): HistoryEntry[] {
  const sorted = history.filter((entry) => isIsoDate(entry.puzzle_date))
    .sort((a, b) => b.puzzle_date.localeCompare(a.puzzle_date));
  const index = sorted.findIndex((entry) => entry.puzzle_date <= date);
  const start = Math.min(
    Math.max(0, sorted.length - 15),
    Math.max(0, (index < 0 ? sorted.length : index) - 7),
  );
  return sorted.slice(start, start + 15);
}

export async function getNearbyHistory(date: string): Promise<HistoryEntry[]> {
  // Navigation is optional: an archive outage must not hide an available solve.
  return nearbyHistory(await getFullHistory(), date);
}
