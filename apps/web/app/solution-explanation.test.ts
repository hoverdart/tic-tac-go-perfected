import { describe, expect, it } from "vitest";
import { buildSolutionExplanation } from "./solution-explanation";

describe("deterministic solution explanations", () => {
  it("derives the goal and a coordinate-based push phase from replay boards", () => {
    const explanation = buildSolutionExplanation(
      [["U", "O", "O", ""], ["", "", "", ""]],
      [["", "U", "O", "O"], ["", "", "", ""]],
      [{ move: "R", board: [["", "U", "O", "O"], ["", "", "", ""]] }],
    );

    expect(explanation.goal).toBe("Finish the horizontal O line from B1 to D1.");
    expect(explanation.phases).toEqual([
      {
        moveIndex: 1,
        title: "Move 1: push the O",
        detail: "Press right to move the O from B1 to D1.",
      },
    ]);
  });
});
