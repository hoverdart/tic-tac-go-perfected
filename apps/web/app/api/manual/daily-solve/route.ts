// POST /api/manual/daily-solve
// Manually triggers the daily board capture and solve pipeline. Intended for
// development and one-off re-runs when the cron job needs to be replayed.
// Accepts POST with no Authorization header check (unlike the cron route), but
// still requires CRON_SECRET to be set so it can authenticate against the
// FastAPI backend. Do not expose this endpoint publicly in production.
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

export async function POST() {
  const cronSecret = process.env.CRON_SECRET;
  const apiBaseUrl = getBackendBaseUrl();

  if (!cronSecret) {
    return NextResponse.json(
      { ok: false, error: "CRON_SECRET is not configured." },
      { status: 500 },
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
    // The FastAPI backend still requires the secret for authentication
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
