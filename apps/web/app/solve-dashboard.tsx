"use client";

import { useEffect, useRef, useState } from "react";
import { Board, type Cell } from "./board";
import { SolvePlayer, type SolveFrame } from "./solve-player";

type SolveStep = { move: string; board: Cell[][] };

type Props = {
  frames: SolveFrame[];
  moves: string | null;
  statesChecked: number | null;
  elapsedMs: number | null;
  parserName: string;
  solverName: string;
  status: "pending" | "solved" | "unsolved" | "failed";
  errorMessage: string | null;
  stepBoards: SolveStep[];
};

const MOVE_ARROWS: Record<string, string> = { D: "↓", U: "↑", L: "←", R: "→" };

function formatMoves(moves: string): string {
  return [...moves].map((c) => MOVE_ARROWS[c] ?? c).join("");
}

function formatElapsed(ms: number | null) {
  if (ms === null) return "Pending";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function statusText(status: Props["status"]) {
  if (status === "solved") return "Solution ready";
  if (status === "unsolved") return "No route found";
  if (status === "failed") return "Capture needs review";
  return "Waiting for the garden to settle";
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
  frames,
  moves,
  statesChecked,
  elapsedMs,
  parserName,
  solverName,
  status,
  errorMessage,
  stepBoards,
}: Props) {
  const hasSteps = stepBoards.length > 0;
  const hasReplay = frames.length > 1;

  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(hasReplay);
  const [showComplete, setShowComplete] = useState(false);
  const stepRefs = useRef<(HTMLButtonElement | null)[]>([]);

  // Playback interval
  useEffect(() => {
    if (!playing || !hasReplay) return;
    const timer = window.setInterval(() => {
      setIndex((cur) => {
        if (cur >= frames.length - 1) {
          setPlaying(false);
          return cur;
        }
        return cur + 1;
      });
    }, 620);
    return () => window.clearInterval(timer);
  }, [playing, hasReplay, frames.length]);

  // Completion flash
  useEffect(() => {
    if (!playing && index >= frames.length - 1 && hasReplay) {
      setShowComplete(true);
      const t = setTimeout(() => setShowComplete(false), 2200);
      return () => clearTimeout(t);
    }
  }, [playing, index, frames.length, hasReplay]);

  // Auto-scroll step library to current card
  useEffect(() => {
    const cardIndex = index - 1; // frame 0 = Start, cards start at frame 1
    const el = stepRefs.current[cardIndex];
    el?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }, [index]);

  function jumpTo(frameIndex: number) {
    setPlaying(false);
    setIndex(frameIndex);
  }

  const displayMoves = moves ? formatMoves(moves) : null;

  return (
    <>
      <section className="focus-area">
        <div className="board-stage">
          <SolvePlayer
            frames={frames}
            index={index}
            playing={playing}
            onSetIndex={setIndex}
            onSetPlaying={setPlaying}
            showComplete={showComplete}
          />
        </div>

        <aside className="solution-panel">
          <div className="panel-heading">
            <p>Today&apos;s path</p>
            <h2 className="moves-display">{displayMoves || "Pending"}</h2>
          </div>

          <dl className="metrics">
            <Metric label="States" value={statesChecked?.toLocaleString() || "Pending"} />
            <Metric label="Solve time" value={formatElapsed(elapsedMs)} />
            <Metric label="Parser" value={parserName} />
            <Metric label="Solver" value={solverName} />
          </dl>

          <div className={`status-note status-${status}`}>
            <span />
            <p>
              {hasSteps
                ? `${stepBoards.length} boards are ready for replay.`
                : errorMessage ||
                  "The daily job will publish the solve path after capture and parsing finish."}
            </p>
          </div>
        </aside>
      </section>

      <section className="steps-section">
        <div className="section-title">
          <div>
            <p>Replay library</p>
            <h2>Every board in the path</h2>
          </div>
          {hasSteps ? <span>{stepBoards.length} saved frames</span> : null}
        </div>

        {hasSteps ? (
          <div className="step-scroll">
            {stepBoards.map((step, i) => {
              const frameIndex = i + 1; // frame 0 is Start
              const isActive = index === frameIndex;
              return (
                <button
                  key={`${i}-${step.move}`}
                  type="button"
                  className={`step-card${isActive ? " step-card-active" : ""}`}
                  ref={(el) => { stepRefs.current[i] = el; }}
                  onClick={() => jumpTo(frameIndex)}
                  aria-label={`Jump to step ${i + 1}: move ${step.move}`}
                  aria-current={isActive ? "true" : undefined}
                >
                  <div className="step-card-header">
                    <span>Step {i + 1}</span>
                    <strong>
                      {MOVE_ARROWS[step.move] ?? step.move}
                    </strong>
                  </div>
                  <Board board={step.board} compact />
                </button>
              );
            })}
          </div>
        ) : (
          <div className="quiet-empty">
            <p>{statusText(status)}</p>
            <span>
              {errorMessage ||
                "Once the daily solve completes, the movement sequence will appear here."}
            </span>
          </div>
        )}
      </section>
    </>
  );
}
