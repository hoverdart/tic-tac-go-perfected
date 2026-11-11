import { ArrowLeft } from "lucide-react";
import { GardenLink } from "./garden-shell";

export function GardenNav({ current }: { current: "guide" | "custom" }) {
  return (
    <nav className="garden-nav" aria-label="Site navigation">
      <GardenLink href="/" className="garden-home"><ArrowLeft size={16} aria-hidden="true" /> Today’s puzzle</GardenLink>
      <GardenLink href={current === "guide" ? "/custom-tic-tac-go-solver" : "/how-to-solve-tic-tac-go"}>
        {current === "guide" ? "Make a board" : "How to play"}
      </GardenLink>
    </nav>
  );
}
