import { buildReplayFrames, type Cell } from "./replay-model";
import { SolvePlayer } from "./solve-player";

export type DailyStatus = "pending" | "solved" | "unsolved" | "failed";

type Props = {
  board: Cell[][] | null;
  moves: string | null;
  statesChecked: number | null;
  elapsedMs: number | null;
  parserName: string;
  solverName: string;
  status: DailyStatus;
  errorMessage: string | null;
  isDemo: boolean;
};

function formatElapsed(ms: number | null): string {
  if (ms === null) return "Pending";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function emptyMessage(status: DailyStatus): string {
  if (status === "failed") return "Capture needs review";
  if (status === "unsolved") return "No route found";
  return "Board pending";
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
export function SolveDashboard({
  board,
  moves,
  statesChecked,
  elapsedMs,
  parserName,
  solverName,
  status,
  errorMessage,
  isDemo,
}: Props) {
  const frames = buildReplayFrames(board, moves);

  return (
    <>
      <div className="wood-stage">
        <SolvePlayer frames={frames} emptyMessage={emptyMessage(status)} />
      </div>

      <details className="diagnostics">
        <summary>Solver details</summary>
        <dl className="metrics">
          <Metric label="States checked" value={statesChecked?.toLocaleString() ?? "Pending"} />
          <Metric label="Solve time" value={formatElapsed(elapsedMs)} />
          <Metric label="Parser" value={parserName} />
          <Metric label="Solver" value={solverName} />
        </dl>
        <p className={`diagnostic-note diagnostic-${status}`}>
          {isDemo
            ? "Demo data is shown locally because API_BASE_URL is not configured."
            : errorMessage ?? (status === "solved" ? "Today's solve path is ready." : "The daily solve is still processing.")}
        </p>
      </details>
    </>
  );
}
