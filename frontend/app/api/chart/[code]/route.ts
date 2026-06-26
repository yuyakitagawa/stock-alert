import { NextRequest, NextResponse } from "next/server";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const SUPABASE_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

const PERIOD_DAYS: Record<string, number> = {
  "1M": 30,
  "3M": 90,
  "6M": 180,
  "1Y": 365,
};

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ code: string }> }
) {
  const { code } = await params;
  const period = req.nextUrl.searchParams.get("period") ?? "1M";
  const days = PERIOD_DAYS[period] ?? 30;

  if (!SUPABASE_URL || !SUPABASE_KEY) {
    return NextResponse.json([]);
  }

  try {
    const cutoff = new Date(Date.now() - days * 86400000).toISOString().split("T")[0];
    const url = `${SUPABASE_URL}/rest/v1/yahoo_price_cache?code=eq.${code}&date=gte.${cutoff}&order=date.asc&select=date,close`;
    const res = await fetch(url, {
      headers: {
        apikey: SUPABASE_KEY,
        Authorization: `Bearer ${SUPABASE_KEY}`,
      },
      next: { revalidate: 3600 },
    });
    if (!res.ok) return NextResponse.json([]);

    const rows: { date: string; close: number | null }[] = await res.json();
    const points = rows
      .filter((r): r is { date: string; close: number } => r.close !== null)
      .map((r) => ({ date: r.date, close: r.close }));

    return NextResponse.json(points);
  } catch {
    return NextResponse.json([]);
  }
}
