import { revalidatePath, revalidateTag } from "next/cache";
import {
  SOLUTION_HISTORY_CACHE_TAG,
  SOLUTION_TODAY_CACHE_TAG,
  solutionDateCacheTag,
} from "../solution-cache";
import { todayIsoDate } from "../solution-data";

function isIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

export function puzzleDateFromPayload(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const puzzleDate = (payload as { puzzle_date?: unknown }).puzzle_date;
  return typeof puzzleDate === "string" && isIsoDate(puzzleDate) ? puzzleDate : null;
}

export function publishSolutionCache(puzzleDate: string): void {
  if (!isIsoDate(puzzleDate)) {
    throw new Error(`Cannot publish an invalid puzzle date: ${puzzleDate}`);
  }

  // The cron Route Handler has just committed new data, so expire these entries
  // immediately. Their next request blocks for one fresh read instead of serving
  // stale content and starting a background refresh.
  revalidateTag(SOLUTION_TODAY_CACHE_TAG, { expire: 0 });
  revalidateTag(SOLUTION_HISTORY_CACHE_TAG, { expire: 0 });
  revalidateTag(solutionDateCacheTag(puzzleDate), { expire: 0 });

  revalidatePath("/");
  revalidatePath("/sitemap.xml");
  revalidatePath("/api/solutions/history");
  revalidatePath(`/solutions/${puzzleDate}`);
  revalidatePath(`/api/solutions/${puzzleDate}`);
}

export function publishSolutionCaches(puzzleDates: string[]): void {
  const uniqueDates = [...new Set(puzzleDates)];
  for (const puzzleDate of uniqueDates) publishSolutionCache(puzzleDate);
}

export function publishOptimizedSolutionCaches(dates: string[]): void {
  if (dates.length === 0) return;
  revalidatePath("/optimized-solutions");
  revalidatePath("/sitemap.xml");
  if (dates.includes(todayIsoDate())) revalidatePath("/");
  for (const date of new Set(dates)) {
    revalidatePath(`/solutions/${date}`);
    revalidatePath(`/solutions/${date}/optimized`);
  }
}

export async function publishDailySolutionCache(date: string, request: Request, secret: string) {
  publishSolutionCache(date);
  const requestOrigin = new URL(request.url).origin;
  const origins = (process.env.SOLUTION_PUBLICATION_ORIGINS ?? "").split(",").filter(Boolean);
  await Promise.all(origins.filter((origin) => origin !== requestOrigin).map(async (origin) => {
    const response = await fetch(`${origin}/api/maintenance/publish-solution-cache`, {
      method: "POST",
      headers: { authorization: `Bearer ${secret}`, "content-type": "application/json" },
      body: JSON.stringify({ dates: [date] }),
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) throw new Error(`Cache publication returned ${response.status} for ${origin}`);
  }));
}
