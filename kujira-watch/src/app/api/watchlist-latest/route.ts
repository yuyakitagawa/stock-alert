import { NextRequest, NextResponse } from "next/server";
import { getLatestDiscDates } from "@/lib/investors";

// /watchlist から叩かれる。ウォッチ中の銘柄コード・投資家名を受け取り、
// それぞれの最新開示日を返す（「保存したあとの新着に気づけない」対策）。
// localStorage保存のため件数はクライアント任せになる。異常な件数のリクエストで
// Supabaseに巨大なIN句を投げないよう上限を掛ける。
const MAX_ITEMS_PER_TYPE = 100;

function parseList(value: string | null): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean)
    .slice(0, MAX_ITEMS_PER_TYPE);
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const stockCodes = parseList(searchParams.get("stocks"));
  const filerNames = parseList(searchParams.get("investors"));
  if (stockCodes.length === 0 && filerNames.length === 0) {
    return NextResponse.json({ stocks: {}, investors: {} });
  }

  try {
    const result = await getLatestDiscDates(stockCodes, filerNames);
    return NextResponse.json(result);
  } catch {
    // 取得失敗時はウォッチリスト表示自体を止めない（最新開示日だけ出さない）
    return NextResponse.json({ stocks: {}, investors: {} });
  }
}
