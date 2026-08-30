"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { HistoryEntry } from "./game-view";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;

function formatShortDate(date: string): string {
  const [, month, day] = date.split("-");
  if (!month || !day) return date;
  return `${MONTHS[Number(month) - 1] ?? month} ${Number(day)}`;
}

function statusIcon(status: HistoryEntry["status"]): string {
  if (status === "solved") return "✓";
  if (status === "failed") return "!";
  return "–";
}

function isHistoryEntry(value: unknown): value is HistoryEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Partial<HistoryEntry>;
  return (
    typeof entry.puzzle_date === "string" &&
    ["pending", "solved", "unsolved", "failed"].includes(entry.status ?? "") &&
    (entry.puzzle_title === null || typeof entry.puzzle_title === "string")
  );
}

type Props = {
  initialHistory: HistoryEntry[];
  currentDate: string;
  isTodayPage: boolean;
  loadSharedHistory: boolean;
};

export function HistoryCarousel({
  initialHistory,
  currentDate,
  isTodayPage,
  loadSharedHistory,
}: Props) {
  const [history, setHistory] = useState(initialHistory);

  useEffect(() => {
    if (!loadSharedHistory) return;

    const controller = new AbortController();
    void fetch("/api/solutions/history", { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("History unavailable"))))
      .then((value: unknown) => {
        if (Array.isArray(value)) setHistory(value.filter(isHistoryEntry));
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          // The solution page remains fully usable if the optional carousel fails.
          setHistory(initialHistory);
        }
      });

    return () => controller.abort();
  }, [initialHistory, loadSharedHistory]);

  if (history.length === 0 && !loadSharedHistory) return null;

  return (
    <nav className="history-carousel" aria-label="Past solutions">
      <div className="history-header">
        <p className="history-label">Past Solutions</p>
        {!isTodayPage && (
          <Link className="history-today-btn" href="/">
            Today ↩
          </Link>
        )}
      </div>
      {history.length > 0 && (
        <div className="history-scroll">
          {history.map((entry) => {
            const isActive = entry.puzzle_date === currentDate;
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
      )}
    </nav>
  );
}
