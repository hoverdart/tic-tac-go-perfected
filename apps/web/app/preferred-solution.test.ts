import { describe, expect, it } from "vitest";
import { preferOptimizedSolution } from "./preferred-solution";
import { optimizedSolutionRecord, parseOptimizedFeed } from "./optimized-solution-data";
import feed from "../data/optimized-v1.json";

const row = parseOptimizedFeed(feed)[0];
const daily = { ...optimizedSolutionRecord(row), solver_name: "push-v4", moves: row.moves + "LR" };

describe("daily best-known replay selection", () => {
  it("selects the shorter verified replay and rebuilds all frames and statistics", () => {
    const chosen = preferOptimizedSolution({ ...daily, states_checked: 999, elapsed_ms: 99 }, row);
    expect(chosen).toEqual(optimizedSolutionRecord(row));
    expect(chosen.moves!.length).toBeLessThan(daily.moves.length);
    expect(chosen.step_boards).toHaveLength(row.moves.length);
    expect(chosen.states_checked).toBeNull();
  });
  it("never replaces a shorter or equal daily replay", () => {
    const shorter = { ...daily, moves: "" };
    expect(preferOptimizedSolution(shorter, row)).toBe(shorter);
    const equal = { ...daily, moves: row.moves };
    expect(preferOptimizedSolution(equal, row)).toBe(equal);
  });
  it("requires matching date and full board geometry and tolerates absent feed rows", () => {
    expect(preferOptimizedSolution(daily, null)).toBe(daily);
    expect(preferOptimizedSolution(daily, { ...row, puzzle_date: "2020-01-01" })).toBe(daily);
    expect(preferOptimizedSolution({ ...daily, board: null }, row).board).toBeNull();
    const different = { ...daily, board: daily.board!.map((line) => [...line]) };
    different.board[0][0] = different.board[0][0] === "B" ? "" : "B";
    expect(preferOptimizedSolution(different, row)).toBe(different);
  });
  it("can recover a failed solve only when its board is known and matches", () => {
    expect(preferOptimizedSolution({ ...daily, status: "failed", moves: null }, row).status).toBe("solved");
  });
});
