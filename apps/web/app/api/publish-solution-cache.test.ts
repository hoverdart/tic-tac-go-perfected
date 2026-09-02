import { beforeEach, describe, expect, it, vi } from "vitest";

const cacheMocks = vi.hoisted(() => ({
  revalidatePath: vi.fn(),
  revalidateTag: vi.fn(),
}));

vi.mock("next/cache", () => cacheMocks);

import {
  publishSolutionCache,
  publishSolutionCaches,
  puzzleDateFromPayload,
} from "./publish-solution-cache";

beforeEach(() => {
  cacheMocks.revalidatePath.mockClear();
  cacheMocks.revalidateTag.mockClear();
});

describe("solution cache publishing", () => {
  it("extracts only valid backend puzzle dates", () => {
    expect(puzzleDateFromPayload({ puzzle_date: "2026-08-30" })).toBe("2026-08-30");
    expect(puzzleDateFromPayload({ puzzle_date: "2026-02-30" })).toBeNull();
    expect(puzzleDateFromPayload({ puzzle_date: 20260830 })).toBeNull();
    expect(puzzleDateFromPayload(null)).toBeNull();
  });

  it("expires shared data and only the newly published date", () => {
    publishSolutionCache("2026-08-30");

    expect(cacheMocks.revalidateTag.mock.calls).toEqual([
      ["solutions:today", { expire: 0 }],
      ["solutions:history", { expire: 0 }],
      ["solutions:date:2026-08-30", { expire: 0 }],
    ]);
    expect(cacheMocks.revalidatePath.mock.calls).toEqual([
      ["/"],
      ["/sitemap.xml"],
      ["/api/solutions/history"],
      ["/solutions/2026-08-30"],
      ["/api/solutions/2026-08-30"],
    ]);
  });

  it("rejects invalid dates before invalidating anything", () => {
    expect(() => publishSolutionCache("not-a-date")).toThrow(
      "Cannot publish an invalid puzzle date",
    );
    expect(cacheMocks.revalidateTag).not.toHaveBeenCalled();
    expect(cacheMocks.revalidatePath).not.toHaveBeenCalled();
  });

  it("publishes each affected date only once after a backfill", () => {
    publishSolutionCaches(["2026-08-31", "2026-08-31", "2026-09-01"]);

    expect(cacheMocks.revalidatePath.mock.calls).toContainEqual(["/sitemap.xml"]);
    expect(cacheMocks.revalidatePath.mock.calls).toContainEqual(["/solutions/2026-08-31"]);
    expect(cacheMocks.revalidatePath.mock.calls).toContainEqual(["/solutions/2026-09-01"]);
    expect(cacheMocks.revalidatePath.mock.calls.filter(([path]) => path === "/solutions/2026-08-31")).toHaveLength(1);
  });
});
