"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

type RefreshState = "idle" | "running" | "done" | "error";

export function ManualRefreshButton() {
  const router = useRouter();
  const [state, setState] = useState<RefreshState>("idle");
  const [message, setMessage] = useState<string | null>(null);

  async function runDailySolve() {
    setState("running");
    setMessage("Capturing today's board...");

    try {
      const response = await fetch("/api/manual/daily-solve", {
        method: "POST",
      });
      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const error = payload?.error || payload?.result?.detail || "Refresh failed.";
        throw new Error(error);
      }

      setState("done");
      setMessage("Solution refreshed.");
      router.refresh();
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Refresh failed.");
    }
  }

  return (
    <div className="manual-refresh">
      <button
        className="refresh-button"
        type="button"
        onClick={runDailySolve}
        disabled={state === "running"}
      >
        <span aria-hidden="true">{state === "running" ? "◌" : "↻"}</span>
        {state === "running" ? "Solving" : "Refresh solve"}
      </button>
      {message ? <p className={`refresh-message refresh-${state}`}>{message}</p> : null}
    </div>
  );
}
