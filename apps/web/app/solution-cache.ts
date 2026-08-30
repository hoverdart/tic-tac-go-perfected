export const SOLUTION_TODAY_CACHE_TAG = "solutions:today";
export const SOLUTION_HISTORY_CACHE_TAG = "solutions:history";

export function solutionDateCacheTag(date: string): string {
  return `solutions:date:${date}`;
}
