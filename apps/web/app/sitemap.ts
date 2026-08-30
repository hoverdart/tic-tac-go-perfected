import type { MetadataRoute } from "next";
import { getSitemapHistory, isIsoDate, todayIsoDate } from "./solution-data";
import { getSiteUrl } from "./site-url";

export const dynamic = "force-static";
export const revalidate = false;

function lastModified(date: string): Date {
  return new Date(`${date}T00:00:00Z`);
}

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const siteUrl = getSiteUrl();
  const history = await getSitemapHistory();
  const homepage: MetadataRoute.Sitemap[number] = {
    url: new URL("/", siteUrl).toString(),
    lastModified: lastModified(todayIsoDate()),
    changeFrequency: "daily",
    priority: 1,
  };

  const solutionPages = history
    .filter((solution) => isIsoDate(solution.puzzle_date))
    .map((solution) => ({
      url: new URL(`/solutions/${solution.puzzle_date}`, siteUrl).toString(),
      lastModified: lastModified(solution.puzzle_date),
      changeFrequency: "never" as const,
      priority: 0.8,
    }));

  return [homepage, ...solutionPages];
}
