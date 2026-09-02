import { NextResponse } from "next/server";
import { publishSolutionCaches } from "../../publish-solution-cache";

export const dynamic = "force-dynamic";

function isIsoDate(value: unknown): value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

export async function POST(request: Request) {
  const secret = process.env.CRON_SECRET;
  if (!secret || request.headers.get("authorization") !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Expected JSON body." }, { status: 400 });
  }
  const dates = body && typeof body === "object" ? (body as { dates?: unknown }).dates : null;
  if (!Array.isArray(dates) || dates.length === 0 || dates.length > 500 || !dates.every(isIsoDate)) {
    return NextResponse.json({ error: "dates must be 1–500 valid ISO dates." }, { status: 400 });
  }
  publishSolutionCaches(dates);
  return NextResponse.json({ ok: true, dates: [...new Set(dates)] });
}
