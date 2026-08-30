import { NextResponse } from "next/server";
import { getFullHistory } from "../../../solution-data";

export const dynamic = "force-static";
export const revalidate = false;

export async function GET() {
  const history = await getFullHistory();
  return NextResponse.json(history, {
    headers: {
      "Cache-Control": "public, max-age=300, s-maxage=86400, stale-while-revalidate=86400",
    },
  });
}
