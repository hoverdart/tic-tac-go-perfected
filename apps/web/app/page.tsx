type Cell = "" | "X" | "O" | "U" | "B";

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

function formatElapsed(ms: number | null) {
  if (ms === null) return "Pending";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function visibleBoard(board: Cell[][] | null) {
  if (!board) return null;

  let lastRow = board.length - 1;
  let lastCol = board[0]?.length ? board[0].length - 1 : 0;

  while (lastRow > 0 && board[lastRow]?.every((cell) => cell === "B")) {
    lastRow -= 1;
  }

  while (
    lastCol > 0 &&
    board.slice(0, lastRow + 1).every((row) => row[lastCol] === "B")
  ) {
    lastCol -= 1;
  }

  return board.slice(0, lastRow + 1).map((row) => row.slice(0, lastCol + 1));
}

function Piece({ cell }: { cell: Cell }) {
  if (cell === "" || cell === "B") return null;
  return <span className={`piece piece-${cell.toLowerCase()}`} aria-label={cell} />;
}

function Board({
  board,
  compact = false,
}: {
  board: Cell[][] | null;
  compact?: boolean;
}) {
  const displayBoard = visibleBoard(board);

  if (!displayBoard) {
    return (
      <div className="board-shell board-empty">
        <div className="empty-board-copy">Board pending</div>
      </div>
    );
  }

  const cols = displayBoard[0]?.length || 1;

  return (
    <div className={compact ? "board-shell board-shell-small" : "board-shell"}>
      <div
        className={compact ? "board-grid board-grid-small" : "board-grid"}
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        aria-label="Tic Tac Go board"
      >
        {displayBoard.flatMap((row, rowIndex) =>
          row.map((cell, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`tile tile-${cell || "empty"}`}
              aria-label={cell || "empty"}
            >
              <Piece cell={cell} />
            </div>
          )),
        )}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export default async function Home() {
  const solution = await getTodaySolution();
  const hasSteps = solution.step_boards.length > 0;
  const board = solution.board ?? sampleBoard;

  return (
    <main className="page">
      <div className="petal-field" aria-hidden="true">
        <span className="petal p1" />
        <span className="petal p2" />
        <span className="petal p3" />
        <span className="petal p4" />
        <span className="petal p5" />
        <span className="petal p6" />
        <span className="petal p7" />
        <span className="petal p8" />
        <span className="petal p9" />
        <span className="petal p10" />
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
        </header>

        <section className="focus-area">
          <div className="board-stage">
            <Board board={board} />
          </div>

          <aside className="solution-panel">
            <div className="panel-heading">
              <p>Today&apos;s path</p>
              <h2>{solution.moves || "Pending"}</h2>
            </div>

            <dl className="metrics">
              <Metric label="States" value={solution.states_checked?.toLocaleString() || "Pending"} />
              <Metric label="Solve time" value={formatElapsed(solution.elapsed_ms)} />
              <Metric label="Parser" value={solution.parser_name} />
              <Metric label="Solver" value={solution.solver_name} />
            </dl>

            <div className={`status-note status-${solution.status}`}>
              <span />
              <p>
                {hasSteps
                  ? `${solution.step_boards.length} boards are ready for replay.`
                  : solution.error_message ||
                    "The daily job will publish the solve path after capture and parsing finish."}
              </p>
            </div>
          </aside>
        </section>

        <section className="steps-section">
          <div className="section-title">
            <p>Replay</p>
            <h2>Step by step</h2>
          </div>

          {hasSteps ? (
            <div className="step-scroll">
              {solution.step_boards.map((step, index) => (
                <article className="step-card" key={`${index}-${step.move}`}>
                  <div className="step-card-header">
                    <span>Step {index + 1}</span>
                    <strong>{step.move}</strong>
                  </div>
                  <Board board={step.board} compact />
                </article>
              ))}
            </div>
          ) : (
            <div className="quiet-empty">
              <p>{statusText(solution.status)}</p>
              <span>
                {solution.error_message ||
                  "Once the daily solve completes, the movement sequence will appear here."}
              </span>
            </div>
          )}
        </section>

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
