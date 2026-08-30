// GET /api/solutions/[date]
// Client-side proxy to the FastAPI backend's GET /solutions/{date} endpoint.
// Called by GameView when the user selects a past date from the history carousel.
// No authentication required — it's a public read of historical solve data.
import { NextResponse } from "next/server";
import { getSolutionByDate } from "../../../solution-data";

export const dynamic = "force-static";
export const dynamicParams = true;
export const revalidate = false;

export function generateStaticParams() {
  return [];
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ date: string }> },
) {
  const { date } = await params;
  const solution = await getSolutionByDate(date);
  if (solution === null) {
    return NextResponse.json({ error: "Solution not found." }, { status: 404 });
  }
  return NextResponse.json(solution, {
    headers: {
      "Cache-Control": "public, max-age=300, s-maxage=31536000, stale-while-revalidate=86400",
    },
  });
}
