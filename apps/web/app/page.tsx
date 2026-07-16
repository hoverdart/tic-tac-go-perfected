import type { Metadata } from "next";
import { GameView } from "./game-view";
import {
  formatLongDate,
  getFullHistory,
  getTodaySolution,
  solutionMetaDescription,
} from "./solution-data";

export const dynamic = "force-dynamic";

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
  const [{ solution, isDemo }, history] = await Promise.all([
    getTodaySolution(),
    getFullHistory(),
  ]);

  return (
    <main className="page">
      <section className="game-scene">
        <GameView
          initialSolution={solution}
          history={history}
          isDemo={isDemo}
          isTodayPage
        />

        <footer className="site-footer">
          <span>Daily board capture and verified replay.</span>
          <span>
            Built by{" "}
            <a href="https://github.com/Abdullah-Waris" target="_blank" rel="noopener noreferrer">Abdullah</a>
            {" & "}
            <a href="https://www.shauryav.com/" target="_blank" rel="noopener noreferrer">Shaurya</a>
          </span>
        </footer>
      </section>
    </main>
  );
}
