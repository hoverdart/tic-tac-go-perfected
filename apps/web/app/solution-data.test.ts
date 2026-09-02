import { describe, expect, it } from "vitest";
import {
  formatLongDate,
  isIsoDate,
  solutionMetaDescription,
} from "./solution-data";
import type { SolutionRecord } from "./game-view";

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
