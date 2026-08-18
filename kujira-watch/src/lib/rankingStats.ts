import type { ArticleContent, DealType } from "@/types/article";

// /ranking/[slug]の集計。記事(microCMS)1件＝EDINET大量保有報告書1件の開示として扱う。
// 「投資家ランキング」のタブ（買い増し・売却・報告書件数）は投資家別に積み上げる。
// 銘柄別に並べると同じタブ内で軸が銘柄に変わってしまい、ランキングの意味が変わるため。
export type RankingSlug = "buys" | "sells" | "filings" | "activist";

// 投資家別ランキングの1行。代表開示（金額が最大の開示）を銘柄・解説記事への内部リンクに使う。
export type FilerRow = {
  filerName: string;
  category?: DealType;
  amount: number;
  count: number;
  topStockCode: string;
  topStockName: string;
  topArticleId: string;
  latestDealDate: string;
};

// 銘柄別ランキング（activist）の1行。
export type StockRow = {
  key: string;
  stockCode: string;
  stockName: string;
  amount: number;
  articleId: string;
  dealDate: string;
  filerName?: string;
  sell: boolean;
};

export function isSell(article: ArticleContent): boolean {
  return (article.tags ?? "").split(",").some((tag) => tag.trim() === "売り");
}

export function buildFilerRows(
  slug: RankingSlug,
  articles: ArticleContent[],
  limit: number
): FilerRow[] {
  // 提出者名が無い過去記事は投資家別に積めないので除外する。
  const target = articles.filter((a) => Boolean(a.filerName));
  const filtered =
    slug === "buys"
      ? target.filter((a) => !isSell(a))
      : slug === "sells"
        ? target.filter(isSell)
        : target;

  const byFiler = new Map<
    string,
    { count: number; total: number; top: ArticleContent; latest: string }
  >();
  for (const a of filtered) {
    const key = a.filerName as string;
    const entry = byFiler.get(key);
    if (!entry) {
      byFiler.set(key, { count: 1, total: a.dealAmount, top: a, latest: a.dealDate });
    } else {
      entry.count += 1;
      entry.total += a.dealAmount;
      if (a.dealAmount > entry.top.dealAmount) entry.top = a;
      if (a.dealDate > entry.latest) entry.latest = a.dealDate;
    }
  }

  return [...byFiler]
    .map(([filerName, { count, total, top, latest }]) => ({
      filerName,
      category: top.dealType,
      amount: total,
      count,
      topStockCode: top.stockCode,
      topStockName: top.stockName,
      topArticleId: top.id,
      latestDealDate: latest,
    }))
    // 件数ランキングは件数優先（同数は合計金額）、買い増し・売却は金額優先（同額は件数）。
    .sort((x, y) =>
      slug === "filings"
        ? y.count - x.count || y.amount - x.amount
        : y.amount - x.amount || y.count - x.count
    )
    .slice(0, limit);
}

export function buildStockRows(articles: ArticleContent[], limit: number): StockRow[] {
  return articles
    .filter((a) => a.dealType === "アクティビスト")
    .sort((x, y) => y.dealAmount - x.dealAmount)
    .slice(0, limit)
    .map((a) => ({
      key: a.id,
      stockCode: a.stockCode,
      stockName: a.stockName,
      amount: a.dealAmount,
      articleId: a.id,
      dealDate: a.dealDate,
      filerName: a.filerName,
      sell: isSell(a),
    }));
}
