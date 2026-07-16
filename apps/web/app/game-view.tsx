import Link from "next/link";
import { SolveDashboard, type DailyStatus } from "./solve-dashboard";
import type { Cell } from "./replay-model";
import { formatLongDate } from "./solution-data";

type SolveStep = {
  move: string;
  board: Cell[][];
};

// Mirrors the full solution record returned by the FastAPI backend.
export type SolutionRecord = {
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
  puzzle_title: string | null;
};

// Lightweight summary used for carousel tiles — no board data.
export type HistoryEntry = {
  puzzle_date: string;
  status: DailyStatus;
  puzzle_title: string | null;
};

type Props = {
  initialSolution: SolutionRecord;
  history: HistoryEntry[];
  isDemo: boolean;
  isTodayPage: boolean;
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;

function formatDate(date: string): string {
  const [year, month, day] = date.split("-");
  if (!year || !month || !day) return date;
  return `${Number(month)}/${Number(day)}/${year}`;
}

function formatShortDate(date: string): string {
  const [, month, day] = date.split("-");
  if (!month || !day) return date;
  return `${MONTHS[Number(month) - 1] ?? month} ${Number(day)}`;
}

function statusText(status: DailyStatus, isDemo: boolean, puzzleTitle: string | null): string {
  if (isDemo) return "Local demo";
  if (status === "solved") return puzzleTitle ?? "Solution ready";
  if (status === "unsolved") return "No route found";
  if (status === "failed") return "Capture needs review";
  return "Solve pending";
}

function statusIcon(status: DailyStatus): string {
  if (status === "solved") return "✓";
  if (status === "failed") return "!";
  return "–";
}

const MOVE_NAMES: Record<string, string> = {
  U: "Up",
  D: "Down",
  L: "Left",
  R: "Right",
};

function SolutionSummary({
  solution,
  isDemo,
  isTodayPage,
}: {
  solution: SolutionRecord;
  isDemo: boolean;
  isTodayPage: boolean;
}) {
  const date = formatLongDate(solution.puzzle_date);
  const subject = isTodayPage
    ? "Today's Tic Tac Go puzzle"
    : `The Tic Tac Go puzzle for ${date}`;
  const namedSubject = solution.puzzle_title
    ? `${subject} "${solution.puzzle_title}"`
    : subject;

  if (solution.status === "solved" && solution.moves !== null) {
    const moveSequence = solution.moves || "Already solved";
    const directions = solution.moves
      .split("")
      .map((move) => MOVE_NAMES[move] ?? move)
      .join(", ");
    return (
      <section className="solution-summary" aria-labelledby="solution-summary-title">
        <h2 id="solution-summary-title">Solution summary</h2>
        <p>
          {namedSubject} was solved in {solution.moves.length} moves. The replay
          above follows the recorded solution from the starting board to the
          completed line.
        </p>
        <p className="solution-moves">
          <strong>Move sequence:</strong> <code>{moveSequence}</code>
          {directions && <span>{directions}</span>}
        </p>
      </section>
    );
  }

  const statusCopy = isDemo
    ? "A local demonstration is shown while the live solution service is unavailable."
    : solution.status === "pending"
      ? "The daily solution is not available yet."
      : solution.status === "unsolved"
        ? "No verified solution was found for this puzzle."
        : "The puzzle capture or solution needs review.";
  return (
    <section className="solution-summary" aria-labelledby="solution-summary-title">
      <h2 id="solution-summary-title">Solution status</h2>
      <p>
        {subject}. {statusCopy}
      </p>
    </section>
  );
}

export function GameView({
  initialSolution,
  history,
  isDemo,
  isTodayPage,
}: Props) {
  const currentSolution = initialSolution;
  const heading = isTodayPage
    ? "Tic Tac Go Solution Today"
    : `Tic Tac Go Solution - ${formatLongDate(currentSolution.puzzle_date)}`;

  return (
    <>
      <header className="game-header">
        <h1 className={isTodayPage ? undefined : "historical-heading"}>{heading}</h1>
        <p>{isTodayPage ? "Daily Solver" : "Verified Replay"}</p>
        <div className="date-pill">
          <time dateTime={currentSolution.puzzle_date}>{formatDate(currentSolution.puzzle_date)}</time>
          <span>{statusText(currentSolution.status, isDemo, currentSolution.puzzle_title)}</span>
        </div>
      </header>

      <SolveDashboard
        board={currentSolution.board}
        moves={currentSolution.moves}
        statesChecked={currentSolution.states_checked}
        elapsedMs={currentSolution.elapsed_ms}
        parserName={currentSolution.parser_name}
        solverName={currentSolution.solver_name}
        status={currentSolution.status}
        errorMessage={currentSolution.error_message}
        isDemo={isDemo}
      />

      <SolutionSummary
        solution={currentSolution}
        isDemo={isDemo}
        isTodayPage={isTodayPage}
      />

      {history.length > 0 && (
        <nav className="history-carousel" aria-label="Past solutions">
          <div className="history-header">
            <p className="history-label">Past Solutions</p>
            {!isTodayPage && (
              <Link className="history-today-btn" href="/">
                Today ↩
              </Link>
            )}
          </div>
          <div className="history-scroll">
            {history.map((entry) => {
              const isActive = entry.puzzle_date === currentSolution.puzzle_date;
              return (
                <Link
                  key={entry.puzzle_date}
                  href={`/solutions/${entry.puzzle_date}`}
                  prefetch={false}
                  className={[
                    "history-tile",
                    `history-tile-${entry.status}`,
                    isActive ? "history-tile-active" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  aria-current={isActive ? "page" : undefined}
                  aria-label={`${entry.puzzle_title ?? formatShortDate(entry.puzzle_date)} — ${entry.status}`}
                >
                  <span className="history-tile-title">
                    {entry.puzzle_title ?? "—"}
                  </span>
                  <span className="history-tile-date">
                    {formatShortDate(entry.puzzle_date)}
                  </span>
                  <span className="history-tile-icon" aria-hidden="true">
                    {statusIcon(entry.status)}
                  </span>
                </Link>
              );
            })}
          </div>
        </nav>
      )}
    </>
  );
}
