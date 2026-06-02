"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { Board } from "./board";
import type { ReplayFrame } from "./replay-model";

const MOVE_ARROWS: Record<string, string> = { D: "↓", U: "↑", L: "←", R: "→" };

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
};

export function SolvePlayer({ frames, emptyMessage }: SolvePlayerProps) {
  const hasReplay = frames.length > 1;
  const reducedMotion = useSyncExternalStore(
    subscribeToReducedMotion,
    prefersReducedMotion,
    serverReducedMotion,
  );
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(hasReplay);
  const activeFrame = frames[index] ?? frames[0] ?? null;
  const effectivePlaying = playing && !reducedMotion;
  const moveCount = Math.max(frames.length - 1, 0);

  useEffect(() => {
    if (!effectivePlaying || !hasReplay) return;
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= frames.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 560);
    return () => window.clearInterval(timer);
  }, [effectivePlaying, frames.length, hasReplay]);

  function step(delta: number) {
    setPlaying(false);
    setIndex((current) => Math.min(Math.max(current + delta, 0), frames.length - 1));
  }

  function replay() {
    setIndex(0);
    setPlaying(hasReplay && !reducedMotion);
  }

  function togglePlayback() {
    if (!hasReplay || reducedMotion) return;
    if (index >= frames.length - 1) setIndex(0);
    setPlaying((current) => !current);
  }

  function scrubTo(frameIndex: number) {
    setPlaying(false);
    setIndex(frameIndex);
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
          disabled={!hasReplay || reducedMotion}
          aria-label={effectivePlaying ? "Pause replay" : "Play replay"}
          title={reducedMotion ? "Playback disabled by reduced motion preference" : effectivePlaying ? "Pause replay" : "Play replay"}
        >
          {effectivePlaying ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={() => step(1)}
          disabled={index >= frames.length - 1 || frames.length === 0}
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
          max={Math.max(moveCount, 1)}
          value={index}
          disabled={!hasReplay}
          onChange={(event) => scrubTo(Number(event.target.value))}
          aria-label="Solution timeline"
        />
        <div className="move-sequence" aria-label={moveCount ? `Solution moves: ${frames.slice(1).map((frame) => frame.move).join("")}` : "Solution pending"}>
          {moveCount ? frames.slice(1).map((frame, frameIndex) => (
            <span key={`${frameIndex}-${frame.move}`} className={frameIndex < index ? "move-complete" : ""}>
              {MOVE_ARROWS[frame.move ?? ""] ?? frame.move}
            </span>
          )) : <span>Awaiting solve path</span>}
        </div>
      </div>
    </section>
  );
}
