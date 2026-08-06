import { NextRequest, NextResponse } from "next/server";
import { searchStocks } from "@/lib/microcms";

// ヘッダーの検索ボックス(StockSearch)から叩かれる。企業名 or 証券コードの部分一致検索。
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q") ?? "";
  if (!q.trim()) {
    return NextResponse.json({ results: [] });
  }

  const results = await searchStocks(q);
  return NextResponse.json({ results });
}
