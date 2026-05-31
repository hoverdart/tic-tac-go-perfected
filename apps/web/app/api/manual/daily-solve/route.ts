import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function POST() {
  const cronSecret = process.env.CRON_SECRET;
  const apiBaseUrl = process.env.API_BASE_URL;

  if (!cronSecret) {
    return NextResponse.json(
      { ok: false, error: "CRON_SECRET is not configured." },
      { status: 500 },
    );
  }

  if (!apiBaseUrl) {
    return NextResponse.json(
      { ok: false, error: "API_BASE_URL is not configured." },
      { status: 500 },
    );
  }

  const response = await fetch(`${apiBaseUrl}/jobs/daily-solve`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${cronSecret}`,
    },
    cache: "no-store",
  });

  const payload = await response.json().catch(() => null);

  return NextResponse.json(
    {
      ok: response.ok,
      backend_status: response.status,
      result: payload,
    },
    { status: response.ok ? 200 : 502 },
  );
}
