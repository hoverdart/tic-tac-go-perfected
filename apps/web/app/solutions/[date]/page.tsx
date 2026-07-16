import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { GameView } from "../../game-view";
import {
  formatLongDate,
  getFullHistory,
  getSolutionByDate,
  isIsoDate,
  solutionMetaDescription,
} from "../../solution-data";

export const dynamic = "force-dynamic";

type Props = {
  params: Promise<{ date: string }>;
};

async function loadSolution(date: string) {
  if (!isIsoDate(date)) notFound();
  const solution = await getSolutionByDate(date);
  if (solution === null) notFound();
  return solution;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { date } = await params;
  const solution = await loadSolution(date);
  const puzzleTitle = solution.puzzle_title ? `: ${solution.puzzle_title}` : "";
  return {
    title: `Tic Tac Go Solution – ${formatLongDate(date)}${puzzleTitle}`,
    description: solutionMetaDescription(solution, false),
    alternates: {
      canonical: `/solutions/${date}`,
    },
  };
}

export default async function HistoricalSolutionPage({ params }: Props) {
  const { date } = await params;
  const [solution, history] = await Promise.all([
    loadSolution(date),
    getFullHistory(),
  ]);

  return (
    <main className="page">
      <section className="game-scene">
        <GameView
          initialSolution={solution}
          history={history}
          isDemo={false}
          isTodayPage={false}
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
