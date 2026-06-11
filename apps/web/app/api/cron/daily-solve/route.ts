// GET /api/cron/daily-solve
// Triggered automatically by Vercel Cron on a daily schedule to kick off the
// board capture and solve pipeline. Vercel passes the CRON_SECRET in the
// Authorization header; we validate it before forwarding the job to the FastAPI
// backend's POST /jobs/daily-solve endpoint. Do not call this manually —
// use /api/manual/daily-solve instead (it accepts POST without the header check).
import { NextResponse } from "next/server";
import { getBackendBaseUrl } from "../../../backend-url";

export const dynamic = "force-dynamic";

// Reads the response body defensively: returns parsed JSON when possible,
// falls back to the first 2 KB of text, or null if the body is empty.
async function readBackendBody(response: Response) {
  const text = await response.text();
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch {
    return text.slice(0, 2_000);
  }
}

export async function GET(request: Request) {
  const cronSecret = process.env.CRON_SECRET;
  const apiBaseUrl = getBackendBaseUrl();
  const authHeader = request.headers.get("authorization");

  if (!cronSecret) {
    return NextResponse.json(
      { ok: false, error: "CRON_SECRET is not configured." },
      { status: 500 },
    );
  }

  // Reject anything that doesn't carry the correct bearer token
  if (authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json(
      { ok: false, error: "Unauthorized" },
      { status: 401 },
    );
  }

  if (!apiBaseUrl) {
    return NextResponse.json(
      { ok: false, error: "Backend URL is not configured." },
      { status: 500 },
    );
  }

  const backendUrl = `${apiBaseUrl}/jobs/daily-solve`;
  let response: Response;

  try {
    // Forward the secret to the backend so it can authenticate the job request
    response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${cronSecret}`,
      },
      cache: "no-store",
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: "Could not reach FastAPI backend.",
        backend_url: backendUrl,
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 502 },
    );
  }

  const payload = await readBackendBody(response);

  return NextResponse.json(
    {
      ok: response.ok,
      backend_status: response.status,
      backend_status_text: response.statusText,
      backend_url: backendUrl,
      result: payload,
    },
    { status: response.ok ? 200 : 502 },
  );
}
