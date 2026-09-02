import { afterEach, describe, expect, it, vi } from "vitest";
import robots from "./robots";
import sitemap, { dynamic, revalidate } from "./sitemap";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("SEO metadata routes", () => {
  it("allows crawling and advertises the production sitemap", () => {
    expect(robots()).toEqual({
      rules: {
        userAgent: "*",
        allow: "/",
      },
      sitemap: "https://tictacgo.shauryav.com/sitemap.xml",
      host: "https://tictacgo.shauryav.com",
    });
  });

  it("uses a configured site origin without retaining a path", () => {
    vi.stubEnv("SITE_URL", "https://preview.example.com/deployment?source=test");

    expect(robots().sitemap).toBe("https://preview.example.com/sitemap.xml");
  });

  it("includes every valid stored solution date", async () => {
    vi.stubEnv("API_BASE_URL", "https://api.example.com");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify([
            {
              puzzle_date: "2026-07-15",
              status: "solved",
              puzzle_title: "Wednesday Puzzle",
            },
            {
              puzzle_date: "2026-07-14",
              status: "solved",
              puzzle_title: "Tuesday Puzzle",
            },
            {
              puzzle_date: "not-a-date",
              status: "failed",
              puzzle_title: null,
            },
          ]),
        ),
      ),
    );

    const entries = await sitemap();

    expect(fetch).toHaveBeenCalledWith(
      "https://api.example.com/solutions/recent?limit=10000",
      {
        cache: "force-cache",
        next: { tags: ["solutions:history"] },
      },
    );
    expect(dynamic).toBe("force-dynamic");
    expect(revalidate).toBe(86400);
    expect(entries.map((entry) => entry.url)).toEqual([
      "https://tictacgo.shauryav.com/",
      "https://tictacgo.shauryav.com/solutions/2026-07-15",
      "https://tictacgo.shauryav.com/solutions/2026-07-14",
      "https://tictacgo.shauryav.com/how-to-solve-tic-tac-go",
      "https://tictacgo.shauryav.com/custom-tic-tac-go-solver",
    ]);
    expect(entries[1]?.lastModified).toEqual(
      new Date("2026-07-15T00:00:00Z"),
    );
  });
});
