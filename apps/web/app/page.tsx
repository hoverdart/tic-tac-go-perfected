import { SolveDashboard, type DailyStatus } from "./solve-dashboard";
import type { Cell } from "./replay-model";

export const dynamic = "force-dynamic";

type SolveStep = {
  move: string;
  board: Cell[][];
};

type SolutionRecord = {
  puzzle_date: string;
  source_url: string;
  parser_name: string;
  solver_name: string;
  board: Cell[][] | null;
  moves: string | null;
  final_board: Cell[][] | null;
  step_boards: SolveStep[];
  states_checked: number | null;
  elapsed_ms: number | null;
  status: DailyStatus;
  error_message: string | null;
};

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
};

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
  const apiBaseUrl = process.env.API_BASE_URL;
  if (!apiBaseUrl) {
    return process.env.NODE_ENV === "development"
      ? { solution: demoSolution, isDemo: true }
      : { solution: unavailableSolution("API_BASE_URL is not configured."), isDemo: false };
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

function formatDate(date: string): string {
  const [year, month, day] = date.split("-");
  if (!year || !month || !day) return date;
  return `${Number(month)}/${Number(day)}/${year}`;
}

function statusText(status: DailyStatus, isDemo: boolean): string {
  if (isDemo) return "Local demo";
  if (status === "solved") return "Solution ready";
  if (status === "unsolved") return "No route found";
  if (status === "failed") return "Capture needs review";
  return "Solve pending";
}

export default async function Home() {
  const { solution, isDemo } = await getTodaySolution();

  return (
    <main className="page">
      <section className="game-scene">
        <header className="game-header">
          <h1>Tic-Tac-Go</h1>
          <p>Daily Solver</p>
          <div className="date-pill">
            <time dateTime={solution.puzzle_date}>{formatDate(solution.puzzle_date)}</time>
            <span>{statusText(solution.status, isDemo)}</span>
          </div>
        </header>

        <SolveDashboard
          board={solution.board}
          moves={solution.moves}
          statesChecked={solution.states_checked}
          elapsedMs={solution.elapsed_ms}
          parserName={solution.parser_name}
          solverName={solution.solver_name}
          status={solution.status}
          errorMessage={solution.error_message}
          isDemo={isDemo}
        />

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
