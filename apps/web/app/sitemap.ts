import type { MetadataRoute } from "next";
import { getSitemapHistory, isIsoDate, todayIsoDate } from "./solution-data";
import { getSiteUrl } from "./site-url";
import { getOptimizedSolutions } from "./optimized-solution-data";

// Sitemap generation must recover if the backend is unavailable during a
// deployment. The history fetch remains tagged/cached; this route simply
// renders a fresh XML response after the authenticated publisher invalidates it.
export const dynamic = "force-dynamic";
export const revalidate = 86400;

function lastModified(date: string): Date {
  return new Date(`${date}T00:00:00Z`);
}

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const siteUrl = getSiteUrl();
  const [history, optimized] = await Promise.all([getSitemapHistory(), getOptimizedSolutions(true)]);
  const homepage: MetadataRoute.Sitemap[number] = {
    url: new URL("/", siteUrl).toString(),
    lastModified: lastModified(todayIsoDate()),
    changeFrequency: "daily",
    priority: 1,
  };

  const solutionPages = history
    .filter((solution) => isIsoDate(solution.puzzle_date) && solution.status !== "pending")
    .map((solution) => ({
      url: new URL(`/solutions/${solution.puzzle_date}`, siteUrl).toString(),
      lastModified: lastModified(solution.puzzle_date),
      changeFrequency: "monthly" as const,
      priority: 0.8,
    }));

  return [
    homepage,
    ...solutionPages,
    { url: new URL("/optimized-solutions", siteUrl).toString(), changeFrequency: "daily", priority: 0.8 },
    ...optimized.map((solution) => ({
      url: new URL(`/solutions/${solution.puzzle_date}/optimized`, siteUrl).toString(),
      lastModified: new Date(solution.updated_at),
      changeFrequency: "monthly" as const,
      priority: 0.8,
    })),
    {
      url: new URL("/how-to-solve-tic-tac-go", siteUrl).toString(),
      changeFrequency: "monthly",
      priority: 0.9,
    },
    {
      url: new URL("/custom-tic-tac-go-solver", siteUrl).toString(),
      changeFrequency: "monthly",
      priority: 0.8,
    },
  ];
}
