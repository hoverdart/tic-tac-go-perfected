import type { Metadata } from "next";
import { CustomBoardMaker } from "../custom-board-maker";

export const metadata: Metadata = {
  title: "Custom Tic Tac Go Solver | Build and Solve a Board",
  description: "Build a Tic Tac Go board and get a verified push-solver replay in seconds.",
  alternates: { canonical: "/custom-tic-tac-go-solver" },
};

export default function CustomSolverPage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: "Custom Tic Tac Go Solver",
    applicationCategory: "GameApplication",
    operatingSystem: "Web",
    description: "Build a Tic Tac Go board and replay a verified push-solver result.",
  };
  return (
    <main className="guide-page">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }} />
      <p className="guide-kicker">Verified push search</p>
      <h1>Custom Tic Tac Go Solver</h1>
      <p className="guide-lead">Make a board, place exactly one player O, then ask the same verified push solver used for the daily replay to find a route.</p>
      <CustomBoardMaker />
      <p className="guide-note">To protect the daily solver, custom searches are limited to 10 new boards per visitor per UTC day and five seconds per board. Repeating an already checked board is free.</p>
    </main>
  );
}
