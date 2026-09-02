import { createHmac } from "node:crypto";
import { NextResponse } from "next/server";
import { getBackendBaseUrl } from "../../backend-url";

export const dynamic = "force-dynamic";

function clientHash(request: Request, secret: string): string {
  const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  const address = forwarded || request.headers.get("x-real-ip") || "unknown";
  return createHmac("sha256", secret).update(address).digest("hex");
}

export async function POST(request: Request) {
  const backendUrl = getBackendBaseUrl();
  const cronSecret = process.env.CRON_SECRET;
  const rateLimitSecret = process.env.CUSTOM_SOLVER_RATE_LIMIT_SECRET;
  if (!backendUrl || !cronSecret || !rateLimitSecret) {
    return NextResponse.json({ error: "Custom solver is not configured." }, { status: 503 });
  }

  const rawBody = await request.text();
  if (!rawBody || rawBody.length > 16_000) {
    return NextResponse.json({ error: "Invalid custom board request." }, { status: 400 });
  }
  try {
    JSON.parse(rawBody);
  } catch {
    return NextResponse.json({ error: "Expected JSON body." }, { status: 400 });
  }

  let response: Response;
  try {
    response = await fetch(`${backendUrl}/solve/custom`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${cronSecret}`,
        "Content-Type": "application/json",
        "X-Custom-Solver-Client-Hash": clientHash(request, rateLimitSecret),
      },
      body: rawBody,
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ error: "Could not reach the custom solver." }, { status: 502 });
  }

  const text = await response.text();
  return new NextResponse(text, {
    status: response.status,
    headers: {
      "Content-Type": response.headers.get("content-type") ?? "application/json",
      ...(response.headers.has("retry-after") ? { "Retry-After": response.headers.get("retry-after")! } : {}),
    },
  });
}
