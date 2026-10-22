import { NextResponse } from "next/server";
import { getSitemapHistory } from "../../../solution-data";

export const dynamic = "force-static";
export const revalidate = 3600;

export async function GET() {
  // A failed refresh must retain the previous archive, not cache an empty one.
  const history = await getSitemapHistory();
  return NextResponse.json(history, {
    headers: {
      "Cache-Control": "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400",
    },
  });
}
