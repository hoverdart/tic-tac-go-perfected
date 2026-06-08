import { NextResponse } from "next/server";
import { getBackendBaseUrl } from "../../../backend-url";

export const dynamic = "force-dynamic";

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
