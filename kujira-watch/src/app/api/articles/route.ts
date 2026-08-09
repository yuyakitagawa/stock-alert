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
  return NextResponse.json({ contents, totalCount });
}
