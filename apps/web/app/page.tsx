import { ManualRefreshButton } from "./manual-refresh-button";
import { SolveDashboard } from "./solve-dashboard";
import type { SolveFrame } from "./solve-player";
import type { Cell } from "./board";

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
  status: "pending" | "solved" | "unsolved" | "failed";
  error_message: string | null;
};

const sampleBoard: Cell[][] = [
  ["X", "", "", "X", "", ""],
  ["", "X", "X", "", "O", ""],
  ["", "O", "", "", "X", "X"],
  ["X", "", "B", "X", "", ""],
  ["", "X", "", "", "", ""],
  ["", "", "", "X", "", "U"],
];

const sampleSolution: SolutionRecord = {
  puzzle_date: new Date().toISOString().slice(0, 10),
  source_url: "",
  parser_name: "gemini",
  solver_name: "bfs",
  board: sampleBoard,
  moves: null,
  final_board: null,
  step_boards: [],
  states_checked: null,
  elapsed_ms: null,
  status: "pending",
  error_message: "Set API_BASE_URL to show today's generated solution.",
};

async function getTodaySolution(): Promise<SolutionRecord> {
  const apiBaseUrl = process.env.API_BASE_URL;
  if (!apiBaseUrl) return sampleSolution;

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/today`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return {
        ...sampleSolution,
        status: "failed",
        error_message: `Backend returned ${response.status}.`,
      };
    }

    return response.json();
  } catch (error) {
    return {
      ...sampleSolution,
      status: "failed",
      error_message: error instanceof Error ? error.message : "Could not reach backend.",
    };
  }
}

function formatDate(date: string) {
  const [year, month, day] = date.split("-");
  if (!year || !month || !day) return date;
  return `${Number(month)}/${Number(day)}/${year}`;
}

function statusText(status: SolutionRecord["status"]) {
  if (status === "solved") return "Solution ready";
  if (status === "unsolved") return "No route found";
  if (status === "failed") return "Capture needs review";
  return "Waiting for the garden to settle";
}

export default async function Home() {
  const solution = await getTodaySolution();
  const board = solution.board ?? sampleBoard;
  const solveFrames: SolveFrame[] = [
    { move: "Start", board },
    ...solution.step_boards,
  ];

  return (
    <main className="page">
      <div className="petal-field" aria-hidden="true">
        <span className="petal p1" />
        <span className="petal p2" />
        <span className="petal p3" />
        <span className="petal p4" />
        <span className="petal p5" />
        <span className="petal p6" />
        <span className="firefly f1" />
        <span className="firefly f2" />
        <span className="firefly f3" />
      </div>

      <section className="scene">
        <header className="hero">
          <h1>Tic-Tac-Go</h1>
          <div className="date-pill">
            <time dateTime={solution.puzzle_date}>{formatDate(solution.puzzle_date)}</time>
            <span>{statusText(solution.status)}</span>
          </div>
          <ManualRefreshButton />
        </header>

        <SolveDashboard
          frames={solveFrames}
          moves={solution.moves}
          statesChecked={solution.states_checked}
          elapsedMs={solution.elapsed_ms}
          parserName={solution.parser_name}
          solverName={solution.solver_name}
          status={solution.status}
          errorMessage={solution.error_message}
          stepBoards={solution.step_boards}
        />

        <footer className="site-footer">
          <p>
            Built by{" "}
            <a href="https://github.com/Abdullah-Waris" target="_blank" rel="noopener noreferrer" className="footer-link">Abdullah</a>
            {" & "}
            <a href="https://www.shauryav.com/" target="_blank" rel="noopener noreferrer" className="footer-link">Shaurya</a>
          </p>
          <span>Daily board capture, parser &amp; solver pipeline.</span>
        </footer>
      </section>
    </main>
  );
}
