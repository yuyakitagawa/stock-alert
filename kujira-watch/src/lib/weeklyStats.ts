import type { ArticleContent, DealType } from "@/types/article";
import { isSellArticle } from "./format";

export type CategoryBreakdown = { dealType: DealType; count: number; amount: number };
export type StockBreakdown = { stockCode: string; stockName: string; count: number; amount: number };
export type FilerBreakdown = { filerName: string; count: number; amount: number };

export type WeeklySummary = {
  totalCount: number;
  totalAmount: number;
  buyCount: number;
  buyAmount: number;
  sellCount: number;
  sellAmount: number;
  // 開示があった投資家分類すべてを金額降順で返す（テーブル表示用。上位3件等への絞り込みは呼び出し側で行わない）。
  categoryBreakdown: CategoryBreakdown[];
};

// /monthly（月別アーカイブ）の「今月の主役」向け。誰が動いたかを金額規模の降順で並べる。
// filerNameはフィールド追加前の記事には無いため、持たない記事は集計から除く。
export function buildFilerRanking(articles: ArticleContent[], limit: number): FilerBreakdown[] {
  const byFiler = new Map<string, FilerBreakdown>();
  for (const article of articles) {
    const filerName = article.filerName?.trim();
    if (!filerName) continue;
    const entry = byFiler.get(filerName) ?? { filerName, count: 0, amount: 0 };
    entry.count += 1;
    entry.amount += article.dealAmount;
    byFiler.set(filerName, entry);
  }
  return [...byFiler.values()].sort((a, b) => b.amount - a.amount).slice(0, limit);
}

// 銘柄別の集計（/monthlyの銘柄ランキング用）。
export function buildStockRanking(articles: ArticleContent[], limit: number): StockBreakdown[] {
  const byStock = new Map<string, StockBreakdown>();
  for (const article of articles) {
    const stock = byStock.get(article.stockCode) ?? {
      stockCode: article.stockCode,
      stockName: article.stockName,
      count: 0,
      amount: 0,
    };
    stock.count += 1;
    stock.amount += article.dealAmount;
    byStock.set(article.stockCode, stock);
  }
  return [...byStock.values()].sort((a, b) => b.amount - a.amount).slice(0, limit);
}

// /weeklyの直近7日サマリー・/monthlyの「今月のポイント」向けに、件数・金額を
// 買い/売り・投資家分類の軸で集計する。
// 記事本文の解説とは別に、数字だけから機械的に導ける事実のみを扱う（解釈・予測はしない）。
export function buildWeeklySummary(articles: ArticleContent[]): WeeklySummary {
  let buyCount = 0;
  let buyAmount = 0;
  let sellCount = 0;
  let sellAmount = 0;
  const categoryMap = new Map<DealType, CategoryBreakdown>();

  for (const article of articles) {
    if (isSellArticle(article.tags)) {
      sellCount += 1;
      sellAmount += article.dealAmount;
    } else {
      buyCount += 1;
      buyAmount += article.dealAmount;
    }

    const category = categoryMap.get(article.dealType) ?? { dealType: article.dealType, count: 0, amount: 0 };
    category.count += 1;
    category.amount += article.dealAmount;
    categoryMap.set(article.dealType, category);
  }

  return {
    totalCount: articles.length,
    totalAmount: buyAmount + sellAmount,
    buyCount,
    buyAmount,
    sellCount,
    sellAmount,
    categoryBreakdown: [...categoryMap.values()].sort((a, b) => b.amount - a.amount),
  };
}
