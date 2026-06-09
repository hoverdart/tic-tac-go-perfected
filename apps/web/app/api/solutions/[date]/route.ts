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
    const response = await fetch(`${apiBaseUrl}/solutions/${date}`, { cache: "no-store" });
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not reach backend." },
      { status: 502 },
    );
  }
}
