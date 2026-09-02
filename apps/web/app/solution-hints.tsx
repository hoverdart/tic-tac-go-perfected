"use client";

import { useState } from "react";
import type { ExplanationPhase } from "./solution-explanation";

export function SolutionHints({ goal, phases }: { goal: string | null; phases: ExplanationPhase[] }) {
  const [revealed, setRevealed] = useState(0);
  if (phases.length === 0) return null;
  return (
    <section className="solution-hints" aria-labelledby="solution-hints-title">
      <h2 id="solution-hints-title">How to solve this board</h2>
      {goal && <p>{goal}</p>}
      <p>Start with the board above, then reveal verified push checkpoints only when you want them.</p>
      {revealed > 0 && (
        <ol>
          {phases.slice(0, revealed).map((phase) => (
            <li key={phase.moveIndex}>
              <strong>{phase.title}.</strong> {phase.detail}
            </li>
          ))}
        </ol>
      )}
      {revealed < phases.length && (
        <button type="button" className="hint-button" onClick={() => setRevealed((value) => value + 1)}>
          Reveal next strategy phase
        </button>
      )}
    </section>
  );
}
