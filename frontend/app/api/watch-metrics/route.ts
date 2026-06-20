import { NextRequest, NextResponse } from "next/server";
import { fetchWatchMetricsMap } from "@/lib/data";

// ウォッチリスト（ブックマーク）銘柄の お得度（52週高値からの下落率）＋ミニチャート＋PER/PBR を返す。
// ブックマークは client(localStorage) 管理のため、コードを受け取って server で Yahoo から取得する。
export async function GET(req: NextRequest) {
  const codesParam = req.nextUrl.searchParams.get("codes") ?? "";
  const codes = codesParam
    .split(",")
    .map((c) => c.trim())
    .filter(Boolean)
    .slice(0, 60); // 過大なリクエストを防ぐ上限

  if (codes.length === 0) return NextResponse.json({});

  try {
    const map = await fetchWatchMetricsMap(codes);
    return NextResponse.json(map);
  } catch {
    return NextResponse.json({});
  }
}
