import type { ArticleContent } from "@/types/article";
import { formatDate, formatDealAmount, isSellArticle } from "./format";
import type { Locale } from "./i18n";

export type StockDealSummary = {
  investorCount: number;
  buyCount: number;
  sellCount: number;
  totalBuyAmount: number;
  totalSellAmount: number;
  firstDealDate: string;
  latestDealDate: string;
  topCategory: string | null;
};

// 銘柄ページの冒頭サマリー用。記事一覧(microCMS)が既に持っている事実（提出者数・
// 取引方向・金額・日付レンジ・投資家分類の内訳）を集計するだけで、新たな創作・LLM呼び出しは
// 行わない。同じ構造のページが銘柄ごとに並ぶ/stocks/[code]の各ページが、集計結果の違いに
// よって内容的に差別化されるようにする狙い（GSC「クロール済み-インデックス未登録」対策）。
export function buildStockDealSummary(articles: ArticleContent[]): StockDealSummary {
  let buyCount = 0;
  let sellCount = 0;
  let totalBuyAmount = 0;
  let totalSellAmount = 0;
  const filerNames = new Set<string>();
  const categoryCounts = new Map<string, number>();

  for (const article of articles) {
    if (isSellArticle(article.tags)) {
      sellCount += 1;
      totalSellAmount += article.dealAmount;
    } else {
      buyCount += 1;
      totalBuyAmount += article.dealAmount;
    }
    if (article.filerName) filerNames.add(article.filerName);
    categoryCounts.set(article.dealType, (categoryCounts.get(article.dealType) ?? 0) + 1);
  }

  const dealDates = articles.map((a) => a.dealDate).sort();
  const topCategory =
    [...categoryCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

  return {
    investorCount: filerNames.size,
    buyCount,
    sellCount,
    totalBuyAmount,
    totalSellAmount,
    firstDealDate: dealDates[0],
    latestDealDate: dealDates[dealDates.length - 1],
    topCategory,
  };
}

export function formatStockDealSummary(
  summary: StockDealSummary,
  stockName: string,
  code: string,
  locale: Locale = "ja"
): string {
  const { investorCount, buyCount, sellCount, totalBuyAmount, totalSellAmount, firstDealDate, latestDealDate, topCategory } =
    summary;

  if (locale === "en") {
    const parts = [`${stockName} (${code}) has ${buyCount + sellCount} large shareholding disclosure(s)`];
    if (investorCount > 0) parts.push(`from ${investorCount} investor(s)`);
    parts.push(`between ${formatDate(firstDealDate, locale)} and ${formatDate(latestDealDate, locale)}.`);
    let text = parts.join(" ");
    if (buyCount > 0) text += ` Buy-side: ${buyCount} (est. ${formatDealAmount(totalBuyAmount, locale)}).`;
    if (sellCount > 0) text += ` Sell-side: ${sellCount} (est. ${formatDealAmount(totalSellAmount, locale)}).`;
    if (topCategory) text += ` Most active category: ${topCategory}.`;
    return text;
  }

  let text = `${stockName}（${code}）は、${formatDate(firstDealDate, locale)}〜${formatDate(latestDealDate, locale)}の間に${
    investorCount > 0 ? `${investorCount}者の投資家から` : ""
  }大量保有・変更報告書が${buyCount + sellCount}件提出されています。`;
  if (buyCount > 0) text += `うち買い方向（取得）が${buyCount}件（推定合計${formatDealAmount(totalBuyAmount, locale)}）`;
  if (sellCount > 0) {
    text += `${buyCount > 0 ? "、" : "うち"}売り方向（譲渡・売却）が${sellCount}件（推定合計${formatDealAmount(totalSellAmount, locale)}）`;
  }
  if (buyCount > 0 || sellCount > 0) text += "です。";
  if (topCategory) text += `提出者の分類では「${topCategory}」による届出が最も多くなっています。`;
  return text;
}
