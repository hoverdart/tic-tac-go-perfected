import type { Metadata } from "next";
import { GameView } from "./game-view";
import { getOptimizedSolutions, matchesOptimizedBoard } from "./optimized-solution-data";
import {
  formatLongDate,
  getFullHistory,
  getTodaySolution,
  solutionMetaDescription,
} from "./solution-data";

export const dynamic = "force-static";
export const revalidate = 300;

export async function generateMetadata(): Promise<Metadata> {
  const { solution } = await getTodaySolution();
  return {
    title: `Tic Tac Go Solution Today – ${formatLongDate(solution.puzzle_date)}`,
    description: solutionMetaDescription(solution, true),
    alternates: {
      canonical: "/",
    },
  };
}

export default async function Home() {
  const [{ solution, isDemo }, history, optimizedSolutions] = await Promise.all([
    getTodaySolution(),
    getFullHistory(),
    getOptimizedSolutions(),
  ]);
  const optimized = optimizedSolutions.find((row) => row.puzzle_date === solution.puzzle_date) ?? null;

  return (
    <main className="page">
      <section className="game-scene">
        <GameView
          initialSolution={solution}
          history={history}
          isDemo={isDemo}
          isTodayPage
          optimized={!isDemo && matchesOptimizedBoard(solution, optimized) && optimized ? { moveCount: optimized.moves.length, optimalityProven: optimized.optimality_proven } : undefined}
        />

        <footer className="site-footer">
          <span>Daily board capture and verified replay.</span>
          <span>
            Built by{" "}
            <a href="https://www.abdullahwaris.com" target="_blank" rel="noopener noreferrer">Abdullah</a>
            {" & "}
            <a href="https://www.shauryav.com/" target="_blank" rel="noopener noreferrer">Shaurya</a>
          </span>
        </footer>
      </section>
    </main>
  );
}
