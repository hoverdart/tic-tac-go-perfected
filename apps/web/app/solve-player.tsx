// Interactive replay controller. Manages frame index, play/pause state, and
// timeline scrubbing. This is a client component because it owns animation
// state and listens to OS-level accessibility preferences in real time.
"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { Board } from "./board";
import type { ReplayFrame } from "./replay-model";

const MOVE_ARROWS: Record<string, string> = { D: "↓", U: "↑", L: "←", R: "→" };

// useSyncExternalStore wiring for the prefers-reduced-motion media query.
// Three arguments:
//   subscribe      — attaches a change listener to the matchMedia object and
//                    returns a cleanup function. Called once on mount.
//   getSnapshot    — returns the current value from window.matchMedia. Called
//                    on every render and after each change event.
//   getServerSnapshot — safe fallback for SSR where window doesn't exist;
//                    always returns false so the server renders the play button
//                    in its active state.
//
// Using useSyncExternalStore (rather than a one-time useEffect) means the
// component re-renders immediately if the user toggles their OS accessibility
// setting mid-session without needing a page reload.
function subscribeToReducedMotion(onChange: () => void): () => void {
  const media = window.matchMedia("(prefers-reduced-motion: reduce)");
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function serverReducedMotion(): boolean {
  return false;
}

type SolvePlayerProps = {
  frames: ReplayFrame[];
  emptyMessage: string;
  hintFirst?: boolean;
};

export function SolvePlayer({ frames, emptyMessage, hintFirst = false }: SolvePlayerProps) {
  const hasReplay = frames.length > 1;
  const reducedMotion = useSyncExternalStore(
    subscribeToReducedMotion,
    prefersReducedMotion,
    serverReducedMotion,
  );
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(hintFirst ? false : hasReplay);
  const [revealed, setRevealed] = useState(hintFirst ? 0 : Math.max(frames.length - 1, 0));
  const activeFrame = frames[index] ?? frames[0] ?? null;

  // When reducedMotion is true the auto-advance timer never starts, so
  // effectivePlaying is false regardless of the play/pause toggle state.
  const maxIndex = Math.min(revealed, Math.max(frames.length - 1, 0));
  const effectivePlaying = playing && !reducedMotion && index < maxIndex;
  const moveCount = Math.max(frames.length - 1, 0);

  // Auto-advance: fires an interval whenever effectivePlaying is true.
  // The updater-function form of setIndex avoids a stale closure over `index` —
  // we always get the latest value without needing `index` in the dep array.
  // The cleanup stops the interval whenever effectivePlaying turns false
  // (pause, reduced motion, or the last frame is reached).
  useEffect(() => {
    if (!effectivePlaying || !hasReplay) return;
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= maxIndex) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 560);
    return () => window.clearInterval(timer);
  }, [effectivePlaying, maxIndex, hasReplay]);

  function step(delta: number) {
    setPlaying(false);
    setIndex((current) => Math.min(Math.max(current + delta, 0), maxIndex));
  }

  // Resets to frame 0 and restarts playback (unless reduced motion is on).
  function replay() {
    setIndex(0);
    setPlaying(hintFirst ? false : hasReplay && !reducedMotion);
  }

  function togglePlayback() {
    if (!hasReplay || reducedMotion) return;
    // If we're at the end, wrap back to the start before resuming
    if (index >= maxIndex) setIndex(0);
    setPlaying((current) => !current);
  }

  function scrubTo(frameIndex: number) {
    setPlaying(false);
    setIndex(Math.min(frameIndex, maxIndex));
  }

  function revealNext() {
    const next = Math.min(revealed + 1, frames.length - 1);
    setRevealed(next);
    setIndex(next);
  }

  function revealAll() {
    setPlaying(false);
    setRevealed(Math.max(frames.length - 1, 0));
  }

  return (
    <section className="replay-player" aria-label="Daily solution replay">
      <div className="board-statusline">
        <span>{activeFrame?.status === "won" ? "Solved" : index === 0 ? "Start" : `Move ${index}`}</span>
        <strong>{index} / {moveCount}</strong>
      </div>

      <Board frame={activeFrame} emptyMessage={emptyMessage} />

      <div className="replay-controls" aria-label="Replay controls">
        <button
          type="button"
          className="icon-button"
          onClick={() => step(-1)}
          disabled={index === 0 || frames.length === 0}
          aria-label="Previous move"
          title="Previous move"
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <button
          type="button"
          className="icon-button replay-primary"
          onClick={togglePlayback}
          disabled={!hasReplay || reducedMotion || maxIndex === 0}
          aria-label={effectivePlaying ? "Pause replay" : "Play replay"}
          title={reducedMotion ? "Playback disabled by reduced motion preference" : effectivePlaying ? "Pause replay" : "Play replay"}
        >
          {effectivePlaying ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={() => step(1)}
          disabled={index >= maxIndex || frames.length === 0}
          aria-label="Next move"
          title="Next move"
        >
          <ChevronRight aria-hidden="true" />
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={replay}
          disabled={!hasReplay}
          aria-label="Replay from start"
          title="Replay from start"
        >
          <RotateCcw aria-hidden="true" />
        </button>
      </div>

      <div className="timeline">
        <input
          type="range"
          min={0}
          max={Math.max(maxIndex, 1)}
          value={index}
          disabled={!hasReplay}
          onChange={(event) => scrubTo(Number(event.target.value))}
          aria-label="Solution timeline"
        />
        {/* Move sequence: each arrow is highlighted once that move has been played */}
        <div className="move-sequence" aria-label={moveCount ? `${index} of ${moveCount} solution moves revealed` : "Solution pending"}>
          {moveCount ? frames.slice(1).map((frame, frameIndex) => (
            <span key={`${frameIndex}-${frame.move}`} className={frameIndex < index ? "move-complete" : ""}>
              {frameIndex < revealed ? (MOVE_ARROWS[frame.move ?? ""] ?? frame.move) : "·"}
            </span>
          )) : <span>Awaiting solve path</span>}
        </div>
      </div>

      {hintFirst && hasReplay && revealed < frames.length - 1 && (
        <div className="hint-controls">
          <button type="button" className="hint-button" onClick={revealNext}>
            Reveal next move
          </button>
          <button type="button" className="hint-button hint-button-secondary" onClick={revealAll}>
            Reveal full solution
          </button>
        </div>
      )}
    </section>
  );
}
