import { afterEach, describe, expect, it, vi } from "vitest";
import { GET } from "./route";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("archive history proxy", () => {
  it("refreshes the complete archive, including dates outside the past year", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    const history = [{ puzzle_date: "2025-01-01", status: "solved", puzzle_title: "Tutorial" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(history))));
    const response = await GET();
    expect(await response.json()).toEqual(history);
    expect(response.headers.get("cache-control")).toContain("s-maxage=3600");
  });

  it("does not cache a successful empty archive during an outage", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Unavailable", { status: 503 })));
    await expect(GET()).rejects.toThrow("503");
  });
});
