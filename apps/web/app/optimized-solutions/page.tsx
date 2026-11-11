import type { Metadata } from "next";
import { GardenLink } from "../garden-shell";
import { GardenNav } from "../garden-nav";
import { getOptimizedSolutions } from "../optimized-solution-data";
import { formatLongDate } from "../solution-data";

export const dynamic = "force-static";
export const revalidate = 300;

export const metadata: Metadata = {
  title: "Tic Tac Go Optimized Solutions",
  description: "Browse verified optimized-v1 solutions for daily Tic Tac Go boards, with move counts, shortest-path status, hints, and replays.",
  alternates: { canonical: "/optimized-solutions" },
};

export default async function OptimizedSolutionsPage() {
  const solutions = await getOptimizedSolutions(true);
  const months = [...new Set(solutions.map((row) => row.puzzle_date.slice(0, 7)))];
  return (
    <main className="guide-page">
      <GardenNav current="optimized" />
      <header className="garden-page-header"><h1>Optimized solutions</h1></header>
      <article className="guide-paper optimized-archive">
        <p className="guide-lead">Explore {solutions.length} verified paths from optimized-v1. Every walk and push counts as one move. “Shortest path verified” means no shorter solution exists; other paths are the best we’ve found so far.</p>
        {months.map((month) => (
          <section key={month}>
            <h2>{new Intl.DateTimeFormat("en-US", { month: "long", year: "numeric", timeZone: "UTC" }).format(new Date(`${month}-01T00:00:00Z`))}</h2>
            <ul>
              {solutions.filter((row) => row.puzzle_date.startsWith(month)).map((row) => (
                <li key={row.puzzle_date}>
                  <GardenLink href={`/solutions/${row.puzzle_date}/optimized`} prefetch={false}>
                    <span>{row.puzzle_title}</span>
                    <time dateTime={row.puzzle_date}>{formatLongDate(row.puzzle_date)}</time>
                    <span>{row.moves.length} moves{row.optimality_proven ? " · shortest path verified" : ""}</span>
                  </GardenLink>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </article>
    </main>
  );
}
