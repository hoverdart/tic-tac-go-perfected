"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { SolveDashboard, type DailyStatus } from "./solve-dashboard";
import type { Cell } from "./replay-model";

type SolveStep = {
  move: string;
  board: Cell[][];
};

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

export type HistoryEntry = {
  puzzle_date: string;
  status: DailyStatus;
  puzzle_title: string | null;
};

type Props = {
  initialSolution: SolutionRecord;
  history: HistoryEntry[];
  isDemo: boolean;
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

export function GameView({ initialSolution, history, isDemo }: Props) {
  const [currentSolution, setCurrentSolution] = useState<SolutionRecord>(initialSolution);
  const [loadingDate, setLoadingDate] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
    }
  }, []);

  const selectDate = useCallback(
    async (date: string) => {
      if (date === currentSolution.puzzle_date || loadingDate !== null) return;

      if (date === initialSolution.puzzle_date) {
        setCurrentSolution(initialSolution);
        return;
      }

      setLoadingDate(date);
      try {
        const res = await fetch(`/api/solutions/${date}`);
        if (res.ok) {
          setCurrentSolution(await res.json());
        }
      } catch {
        // keep current solution on network error
      } finally {
        setLoadingDate(null);
      }
    },
    [currentSolution.puzzle_date, initialSolution, loadingDate],
  );

  const showDemo = isDemo && currentSolution.puzzle_date === initialSolution.puzzle_date;

  return (
    <>
      <header className="game-header">
        <h1>Tic-Tac-Go</h1>
        <p>Daily Solver</p>
        <div className="date-pill">
          <time dateTime={currentSolution.puzzle_date}>{formatDate(currentSolution.puzzle_date)}</time>
          <span>{statusText(currentSolution.status, showDemo, currentSolution.puzzle_title)}</span>
        </div>
      </header>

      <SolveDashboard
        key={currentSolution.puzzle_date}
        board={currentSolution.board}
        moves={currentSolution.moves}
        statesChecked={currentSolution.states_checked}
        elapsedMs={currentSolution.elapsed_ms}
        parserName={currentSolution.parser_name}
        solverName={currentSolution.solver_name}
        status={currentSolution.status}
        errorMessage={currentSolution.error_message}
        isDemo={showDemo}
      />

      {history.length > 0 && (
        <nav className="history-carousel" aria-label="Past solutions">
          <p className="history-label">Past Solutions</p>
          <div className="history-scroll" ref={scrollRef}>
            {[...history].reverse().map((entry) => {
              const isActive = entry.puzzle_date === currentSolution.puzzle_date;
              const isLoading = entry.puzzle_date === loadingDate;
              return (
                <button
                  key={entry.puzzle_date}
                  type="button"
                  className={[
                    "history-tile",
                    `history-tile-${entry.status}`,
                    isActive ? "history-tile-active" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  onClick={() => void selectDate(entry.puzzle_date)}
                  disabled={loadingDate !== null}
                  aria-pressed={isActive}
                  aria-label={`${entry.puzzle_title ?? formatShortDate(entry.puzzle_date)} — ${entry.status}`}
                >
                  <span className="history-tile-date">
                    {entry.puzzle_title ?? formatShortDate(entry.puzzle_date)}
                  </span>
                  <span className="history-tile-icon" aria-hidden="true">
                    {isLoading ? "…" : statusIcon(entry.status)}
                  </span>
                </button>
              );
            })}
          </div>
        </nav>
      )}
    </>
  );
}
