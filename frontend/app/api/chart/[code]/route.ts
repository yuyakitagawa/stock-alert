import { NextRequest, NextResponse } from "next/server";

const RANGE_MAP: Record<string, { range: string; interval: string }> = {
  "1M": { range: "1mo",  interval: "1d"  },
  "3M": { range: "3mo",  interval: "1d"  },
  "6M": { range: "6mo",  interval: "1wk" },
  "1Y": { range: "1y",   interval: "1wk" },
};

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ code: string }> }
) {
  const { code } = await params;
  const period = req.nextUrl.searchParams.get("period") ?? "1M";
  const { range, interval } = RANGE_MAP[period] ?? RANGE_MAP["1M"];

  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${code}.T?range=${range}&interval=${interval}`;

  try {
    const res = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; StockSignal/1.0)" },
      next: { revalidate: 3600 },
    });
    if (!res.ok) return NextResponse.json([], { status: 200 });

    const data = await res.json();
    const result = data?.chart?.result?.[0];
    if (!result) return NextResponse.json([]);

    const timestamps: number[] = result.timestamp ?? [];
    const closes: (number | null)[] = result.indicators?.quote?.[0]?.close ?? [];

    const points = timestamps
      .map((t, i) => ({
        date: new Date(t * 1000).toISOString().split("T")[0],
        close: closes[i],
      }))
      .filter((d): d is { date: string; close: number } => d.close !== null);

    return NextResponse.json(points);
  } catch {
    return NextResponse.json([]);
  }
}
