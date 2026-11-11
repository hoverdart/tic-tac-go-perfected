import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { GameView } from "../../../game-view";
import { formatLongDate, nearbyHistory } from "../../../solution-data";
import { getOptimizedSolution, getOptimizedSolutions, optimizedSolutionRecord } from "../../../optimized-solution-data";

export const dynamic = "force-static";
export const dynamicParams = true;
export const revalidate = 300;

export function generateStaticParams() {
  // The feed grows independently of deployments; generate and cache on demand.
  return [];
}

type Props = { params: Promise<{ date: string }> };

async function load(date: string) {
  const row = await getOptimizedSolution(date);
  if (!row) notFound();
  return row;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { date } = await params;
  const row = await load(date);
  return {
    title: `Tic Tac Go Optimized Solution – ${formatLongDate(date)}: ${row.puzzle_title}`,
    description: `Replay the ${row.moves.length}-move optimized solution for ${row.puzzle_title}, ${formatLongDate(date)}, from optimized-v1. ${row.optimality_proven ? "Verified globally shortest path." : "Verified best-known path with step-by-step hints."}`,
    alternates: { canonical: `/solutions/${date}/optimized` },
    robots: { index: true, follow: true },
  };
}

export default async function OptimizedSolutionPage({ params }: Props) {
  const { date } = await params;
  const row = await load(date);
  const history = (await getOptimizedSolutions()).map((solution) => ({
    puzzle_date: solution.puzzle_date, puzzle_title: solution.puzzle_title, status: "solved" as const,
  }));
  return (
    <main className="page">
      <section className="game-scene">
        <GameView initialSolution={optimizedSolutionRecord(row)} history={nearbyHistory(history, date)} isDemo={false}
          isTodayPage={false} isOptimizedPage
          optimized={{ moveCount: row.moves.length, optimalityProven: row.optimality_proven }} />
        <footer className="site-footer"><span>Locally optimized. Independently replay-verified.</span></footer>
      </section>
    </main>
  );
}
