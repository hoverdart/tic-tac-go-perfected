// GET /api/solutions/[date]
// Client-side proxy to the FastAPI backend's GET /solutions/{date} endpoint.
// Called by GameView when the user selects a past date from the history carousel.
// No authentication required — it's a public read of historical solve data.
import { NextResponse } from "next/server";
import { getBackendBaseUrl } from "../../../backend-url";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ date: string }> },
) {
  const { date } = await params;
  const apiBaseUrl = getBackendBaseUrl();

  if (!apiBaseUrl) {
    return NextResponse.json({ error: "Backend URL is not configured." }, { status: 500 });
  }

  try {
    const response = await fetch(`${apiBaseUrl}/solutions/${date}`, {
      next: { revalidate: 3600 },
    });
    const data = await response.json();
    return NextResponse.json(data, {
      status: response.status,
      headers: {
        "Cache-Control": "public, max-age=300, s-maxage=3600, stale-while-revalidate=3600",
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not reach backend." },
      { status: 502 },
    );
  }
}
