import { afterEach, describe, expect, it, vi } from "vitest";
import { getOptimizedSolution, getOptimizedSolutions, matchesOptimizedBoard, mergeOptimizedSolutions, optimizedSolutionRecord, parseOptimizedFeed } from "./optimized-solution-data";
import fallbackFeed from "../data/optimized-v1.json";

afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });

function fixture() {
  const feed = structuredClone(fallbackFeed);
  feed.solutions = [feed.solutions[0]];
  return feed;
}

describe("optimized-v1 publication", () => {
  it("independently replays the entire bundled backfill", () => {
    const rows = parseOptimizedFeed(fallbackFeed);
    expect(rows.length).toBeGreaterThanOrEqual(414);
    expect(rows.every((row) => optimizedSolutionRecord(row).status === "solved")).toBe(true);
  });

  it("rejects moves that do not solve, duplicate dates, loss states, and overstated proofs", () => {
    const empty = fixture(); empty.solutions[0].moves = "";
    expect(() => parseOptimizedFeed(empty)).toThrow();
    const blocked = fixture(); blocked.solutions[0].moves = "U" + blocked.solutions[0].moves;
    expect(() => parseOptimizedFeed(blocked)).toThrow();
    const duplicate = fixture(); duplicate.solutions.push(duplicate.solutions[0]);
    expect(() => parseOptimizedFeed(duplicate)).toThrow();
    const badProof = fixture(); badProof.solutions[0].lower_bound++;
    expect(() => parseOptimizedFeed(badProof)).toThrow();
    const xLoss = fixture(); xLoss.solutions[0].board[0] = ["X", "X", "X", "B", "B", "B", "B", "B"];
    expect(() => parseOptimizedFeed(xLoss)).toThrow();
  });

  it("keeps the shorter path and filters future dates", () => {
    const [row] = parseOptimizedFeed(fixture());
    const longer = { ...row, moves: row.moves + "R", optimality_proven: false };
    const future = { ...row, puzzle_date: "2099-01-01" };
    expect(mergeOptimizedSolutions([row], [longer, future])).toEqual([row]);
    expect(mergeOptimizedSolutions([row], [])).toEqual([]);
    expect(() => mergeOptimizedSolutions([row], [{ ...longer, optimality_proven: true }])).toThrow("certificate");
  });

  it("links only matching dated board geometry", () => {
    const [row] = parseOptimizedFeed(fixture());
    const daily = optimizedSolutionRecord(row);
    expect(matchesOptimizedBoard(daily, row)).toBe(true);
    expect(matchesOptimizedBoard({ ...daily, board: null }, row)).toBe(false);
    expect(matchesOptimizedBoard({ ...daily, puzzle_date: "2020-01-01" }, row)).toBe(false);
    expect(matchesOptimizedBoard(daily, null)).toBe(false);
  });

  it("uses bundled verified results during an outage and preserves missing-date ISR pages", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    expect((await getOptimizedSolutions()).length).toBeGreaterThanOrEqual(414);
    expect(await getOptimizedSolution("2025-01-01")).not.toBeNull();
    await expect(getOptimizedSolution("2025-01-02")).rejects.toThrow("offline");
    await expect(getOptimizedSolutions(true)).rejects.toThrow("offline");
    expect(await getOptimizedSolution("2099-01-01")).toBeNull();
  });

  it("accepts independently published new dates without a web deployment", async () => {
    const feed = fixture();
    feed.solutions[0].puzzle_date = "2025-01-02";
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => feed }));
    expect(await getOptimizedSolution("2025-01-02")).toMatchObject({ puzzle_date: "2025-01-02" });
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("optimized-v1-data"), expect.objectContaining({ next: { revalidate: 300, tags: ["optimized-v1"] } }));
    expect(await getOptimizedSolution("2025-01-03")).toBeNull();
  });
});
