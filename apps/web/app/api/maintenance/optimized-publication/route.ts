import { timingSafeEqual } from "node:crypto";
import { getBackendBaseUrl } from "../../../backend-url";
import { isIsoDate } from "../../../solution-data";
import { publishOptimizedSolutionCaches } from "../../publish-solution-cache";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const secret = process.env.OPTIMIZED_PUBLICATION_SECRET;
  const expected = Buffer.from(`Bearer ${secret ?? ""}`);
  const provided = Buffer.from(request.headers.get("authorization") ?? "");
  if (!secret || provided.length !== expected.length || !timingSafeEqual(provided, expected)) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "Expected JSON body." }, { status: 400 });
  }
  if (!payload || typeof payload !== "object") {
    return Response.json({ error: "Expected publication request." }, { status: 400 });
  }
  const { phase, revision, dates } = payload as { phase?: unknown; revision?: unknown; dates?: unknown };
  if (phase === "refresh") {
    if (!Array.isArray(dates) || dates.length > 50 ||
        !dates.every((date) => typeof date === "string" && isIsoDate(date))) {
      return Response.json({ error: "Invalid publication dates." }, { status: 400 });
    }
    publishOptimizedSolutionCaches(dates);
    return Response.json({ ok: true });
  }
  if ((phase !== "import" && phase !== "ack") ||
      typeof revision !== "string" || !/^[0-9a-f]{40}$/.test(revision)) {
    return Response.json({ error: "Invalid publication revision." }, { status: 400 });
  }
  const backend = getBackendBaseUrl();
  const cronSecret = process.env.CRON_SECRET;
  if (!backend || !cronSecret) return Response.json({ error: "Publication is unavailable." }, { status: 503 });

  try {
    const response = await fetch(`${backend}/jobs/optimized-publication${phase === "ack" ? "/ack" : ""}`, {
      method: "POST",
      headers: { authorization: `Bearer ${cronSecret}`, "content-type": "application/json" },
      body: JSON.stringify({ revision }),
      cache: "no-store",
      signal: AbortSignal.timeout(25_000),
    });
    if (response.ok) return Response.json(await response.json());
    console.error("Optimized publication backend status", response.status);
  } catch (error) {
    console.error("Optimized publication backend request failed", error);
  }
  return Response.json({ error: "Backend publication failed." }, { status: 502 });
}
