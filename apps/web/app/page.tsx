// Next.js Server Component — fetches today's solution and recent history from
// the FastAPI backend, then passes them to GameView for rendering. Static
// caching is disabled so the board is always the current day's data on every
// request (not a stale build-time snapshot).
import type { Metadata } from "next";
import { GameView, type SolutionRecord, type HistoryEntry } from "./game-view";
import { getBackendBaseUrl } from "./backend-url";

export const dynamic = "force-dynamic";

// Hardcoded demo board used in two situations:
//   1. Local development when no backend URL is configured (isDemo: true)
//   2. As a structural template for unavailableSolution() — its board shape
//      keeps the UI intact even when real data is missing.
const demoSolution: SolutionRecord = {
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

// Spreads the demo board shape so the UI doesn't break when the backend is
// unavailable — only the data fields are replaced with the error context.
function unavailableSolution(errorMessage: string): SolutionRecord {
  return {
    ...demoSolution,
    board: null,
    moves: null,
    states_checked: null,
    elapsed_ms: null,
    status: "failed",
    error_message: errorMessage,
  };
}

async function getTodaySolution(): Promise<{ solution: SolutionRecord; isDemo: boolean }> {
  const apiBaseUrl = getBackendBaseUrl();

  // In local dev with no backend configured, show the demo board so the UI is
  // fully interactive. In production, surface the error instead.
  if (!apiBaseUrl) {
    return process.env.NODE_ENV === "development"
      ? { solution: demoSolution, isDemo: true }
      : { solution: unavailableSolution("Backend URL is not configured."), isDemo: false };
  }

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/today`, { cache: "no-store" });
    if (!response.ok) {
      return { solution: unavailableSolution(`Backend returned ${response.status}.`), isDemo: false };
    }
    return { solution: await response.json(), isDemo: false };
  } catch (error) {
    return {
      solution: unavailableSolution(error instanceof Error ? error.message : "Could not reach backend."),
      isDemo: false,
    };
  }
}

// Returns an empty array on any failure — the carousel just doesn't render
// rather than breaking the page.
async function getRecentHistory(): Promise<HistoryEntry[]> {
  const apiBaseUrl = getBackendBaseUrl();
  if (!apiBaseUrl) return [];

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/recent?limit=30`, { cache: "no-store" });
    if (!response.ok) return [];
    return await response.json();
  } catch {
    return [];
  }
}

function formatDate(date: string): string {
  const [year, month, day] = date.split("-");
  if (!year || !month || !day) return date;
  return `${Number(month)}/${Number(day)}/${year}`;
}

// Exported separately from the page component so Next.js can call it during
// the request to populate <head> tags. This overrides the fallback title set
// in layout.tsx with the current date.
export async function generateMetadata(): Promise<Metadata> {
  const today = new Date().toISOString().slice(0, 10);
  return {
    title: `Tic-Tac-Go Solution | ${formatDate(today)}`,
  };
}

export default async function Home() {
  const [{ solution, isDemo }, history] = await Promise.all([
    getTodaySolution(),
    getRecentHistory(),
  ]);

  return (
    <main className="page">
      <section className="game-scene">
        <GameView initialSolution={solution} history={history} isDemo={isDemo} />

        <footer className="site-footer">
          <span>Daily board capture and optimal replay.</span>
          <span>
            Built by{" "}
            <a href="https://github.com/Abdullah-Waris" target="_blank" rel="noopener noreferrer">Abdullah</a>
            {" & "}
            <a href="https://www.shauryav.com/" target="_blank" rel="noopener noreferrer">Shaurya</a>
          </span>
        </footer>
      </section>
    </main>
  );
}
