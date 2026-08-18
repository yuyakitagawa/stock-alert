import { NextRequest, NextResponse } from "next/server";
import { getArticleList } from "@/lib/microcms";
import { DEAL_TYPES, type DealType } from "@/types/article";

// 記事一覧のオートスクロール（無限スクロール）用。InfiniteArticleListから叩かれる。
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const offset = Number(searchParams.get("offset")) || 0;
  const dealTypeParam = searchParams.get("dealType");
  const dealType = (DEAL_TYPES as string[]).includes(dealTypeParam ?? "")
    ? (dealTypeParam as DealType)
    : undefined;
  const translatedOnly = searchParams.get("translatedOnly") === "1";

  const { contents, totalCount } = await getArticleList({ offset, dealType, translatedOnly });
  return NextResponse.json(
    { contents, totalCount },
    {
      headers: {
        // CDN(Vercel Edge)にクエリ文字列ごとキャッシュさせる。記事ページのISR(60s)と同じ鮮度。
        // s-maxage=CDN側TTL、stale-while-revalidate=期限切れ後も古い応答を返しつつ裏で再取得する猶予。
        "Cache-Control": "public, s-maxage=60, stale-while-revalidate=300",
      },
    },
  );
}
