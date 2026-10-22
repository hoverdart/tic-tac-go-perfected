import { describe, expect, it } from "vitest";
import {
  applyReplayMove,
  buildReplayFrames,
  createReplayStart,
  replayStatus,
  visibleBoard,
  type Cell,
} from "./replay-model";

function board(rows: string[]): Cell[][] {
  return rows.map((row) => [...row].map((cell) => cell === "." ? "" : cell as Cell));
}

describe("replay model", () => {
  it("renders catalog borders like screenshot padding and replays the corrected October 21 board", () => {
    const catalog = board([
      "BBBBBBBB", "B.X..X.B", "B..X.XXB", "B..BO..B",
      "BX.OUX.B", "B.X.XX.B", "B.XX...B", "BBBBBBBB",
    ]);
    const visible = visibleBoard(catalog);
    expect(visible).toHaveLength(6);
    expect(visible[0]).toHaveLength(6);
    expect(visible[2][2]).toBe("B");
    const invalid = buildReplayFrames(catalog, "RULUL");
    expect(invalid.at(-1)?.status).not.toBe("won");
    const valid = buildReplayFrames(catalog, "UULRDDRDUULULLRRDDRDLUUULLDDRLUURRDRDDLULLUURR");
    expect(valid.at(-1)?.status).toBe("won");
  });

  it("walks the user piece while preserving its stable id", () => {
    const start = createReplayStart(board(["U..", "...", "..."]));
    const next = applyReplayMove(start, "R");

    expect(next.board).toEqual(board([".U.", "...", "..."]));
    expect(next.pieces.find((piece) => piece.kind === "U")).toMatchObject({
      id: "u-1",
      row: 0,
      col: 1,
    });
  });

  it("pushes an X piece into an empty cell", () => {
    const next = applyReplayMove(createReplayStart(board(["UX.", "...", "..."])), "R");

    expect(next.board).toEqual(board([".UX", "...", "..."]));
    expect(next.pieces.find((piece) => piece.kind === "X")).toMatchObject({
      id: "x-1",
      row: 0,
      col: 2,
    });
  });

  it("pushes an O piece and keeps the same piece id in later frames", () => {
    const frames = buildReplayFrames(board(["UO.", "...", "..."]), "RDLU");
    const pushedO = frames[1].pieces.find((piece) => piece.kind === "O");
    const finalO = frames.at(-1)?.pieces.find((piece) => piece.kind === "O");

    expect(pushedO).toMatchObject({ id: "o-1", row: 0, col: 2 });
    expect(finalO?.id).toBe("o-1");
  });

  it("does not move through barriers or push into occupied cells", () => {
    const barrier = applyReplayMove(createReplayStart(board(["UB.", "...", "..."])), "R");
    const blockedPush = applyReplayMove(createReplayStart(board(["UXO", "...", "..."])), "R");

    expect(barrier.board).toEqual(board(["UB.", "...", "..."]));
    expect(blockedPush.board).toEqual(board(["UXO", "...", "..."]));
  });

  it("recognizes useful-piece wins and X losses", () => {
    expect(replayStatus(board(["UOO", "...", "..."]))).toBe("won");
    expect(replayStatus(board(["XXX", ".U.", "..."]))).toBe("lost");
  });

  it("crops only trailing rows and columns that are entirely barriers", () => {
    expect(visibleBoard(board(["UO.B", "...B", "BBBB", "BBBB"]))).toEqual(
      board(["UO.", "..."]),
    );
  });

  it("builds deterministic frames that match expected stored step boards", () => {
    const frames = buildReplayFrames(board(["UO.", "...", "..."]), "RDL");

    expect(frames.map((frame) => frame.board)).toEqual([
      board(["UO.", "...", "..."]),
      board([".UO", "...", "..."]),
      board(["..O", ".U.", "..."]),
      board(["..O", "U..", "..."]),
    ]);
  });
});
