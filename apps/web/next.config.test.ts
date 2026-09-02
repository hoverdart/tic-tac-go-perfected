import { describe, expect, it } from "vitest";
import config from "./next.config";

describe("canonical-domain redirects", () => {
  it("permanently redirects every legacy-domain path to the canonical equivalent", async () => {
    const redirects = await config.redirects?.();

    expect(redirects).toContainEqual({
      source: "/:path*",
      has: [{ type: "host", value: "tictacgo.abdullahwaris.com" }],
      destination: "https://tictacgo.shauryav.com/:path*",
      permanent: true,
    });
  });
});
