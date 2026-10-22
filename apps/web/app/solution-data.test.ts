import { afterEach, describe, expect, it, vi } from "vitest";
import {
  formatLongDate,
  isIsoDate,
  solutionMetaDescription,
  getSolutionByDate,
  nearbyHistory,
  getNearbyHistory,
  getFullHistory,
} from "./solution-data";
import type { SolutionRecord } from "./game-view";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

const solvedRecord: SolutionRecord = {
  puzzle_date: "2026-07-15",
  source_url: "",
  parser_name: "gemini",
  solver_name: "push-v3",
  board: [["U", "O", "O"]],
  moves: "RULD",
  final_board: [["U", "O", "O"]],
  step_boards: [],
  states_checked: 42,
  elapsed_ms: 3,
  status: "solved",
  error_message: null,
  puzzle_title: "Test Puzzle",
};

describe("solution SEO helpers", () => {
  it("validates real ISO calendar dates", () => {
    expect(isIsoDate("2026-07-15")).toBe(true);
    expect(isIsoDate("2026-02-30")).toBe(false);
    expect(isIsoDate("July-15-2026")).toBe(false);
  });

  it("formats dates consistently in UTC", () => {
    expect(formatLongDate("2026-07-15")).toBe("July 15, 2026");
  });

  it("describes the dated solution and move count", () => {
    expect(solutionMetaDescription(solvedRecord, false)).toBe(
      "See the July 15, 2026 Tic Tac Go solution for Test Puzzle, including verified hint-first steps and the 4-move replay.",
    );
  });
});

describe("historical page availability", () => {
  it.each([429, 500, 503])("does not turn HTTP %s into a noindex/404 result", async (status) => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Unavailable", { status })));
    await expect(getSolutionByDate("2026-07-15")).rejects.toThrow(String(status));
  });

  it("preserves network errors so cached good pages survive failed regeneration", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await expect(getSolutionByDate("2026-07-15")).rejects.toThrow("offline");
  });

  it("still identifies genuinely missing dates", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Missing", { status: 404 })));
    await expect(getSolutionByDate("2026-07-15")).resolves.toBeNull();
    await expect(getSolutionByDate("2026-02-30")).resolves.toBeNull();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("returns the requested puzzle with bounded cache freshness", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(solvedRecord))));
    await expect(getSolutionByDate(solvedRecord.puzzle_date)).resolves.toEqual(solvedRecord);
    expect(fetch).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({
      next: { tags: ["solutions:date:2026-07-15"], revalidate: 3600 },
    }));
  });

  it("rejects a mismatched puzzle instead of creating duplicate dated pages", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(solvedRecord))));
    await expect(getSolutionByDate("2026-07-14")).rejects.toThrow("date mismatch");
  });

  it("includes older neighbors outside the recent-history window", () => {
    const history = Array.from({ length: 500 }, (_, index) => ({
      puzzle_date: new Date(Date.UTC(2025, 0, index + 1)).toISOString().slice(0, 10),
      puzzle_title: null,
      status: "solved" as const,
    }));
    const neighbors = nearbyHistory(history, "2025-01-15");
    expect(neighbors).toHaveLength(15);
    expect(neighbors.map((entry) => entry.puzzle_date)).toContain("2025-01-14");
    expect(neighbors.map((entry) => entry.puzzle_date)).toContain("2025-01-16");
    expect(nearbyHistory([], "2025-01-15")).toEqual([]);
    expect(nearbyHistory(history, history[0].puzzle_date)).toHaveLength(15);
    expect(nearbyHistory(history, history[499].puzzle_date)).toHaveLength(15);
  });

  it("retains dates older than a year when the carousel refreshes", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    const history = [{ puzzle_date: "2025-01-01", status: "solved", puzzle_title: "Tutorial" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(history))));
    await expect(getFullHistory()).resolves.toEqual(history);
    expect(fetch).toHaveBeenCalledWith(
      "https://api.example.com/solutions/recent?limit=10000", expect.any(Object),
    );
  });

  it("keeps an available solution usable when optional history is down", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url: string) => Promise.resolve(
      url.includes("/recent")
        ? new Response("Unavailable", { status: 503 })
        : new Response(JSON.stringify(solvedRecord)),
    )));
    await expect(getSolutionByDate(solvedRecord.puzzle_date)).resolves.toEqual(solvedRecord);
    await expect(getNearbyHistory(solvedRecord.puzzle_date)).resolves.toEqual([]);
  });
});
