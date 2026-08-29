import type { ArticleContent } from "@/types/article";
import type { StockFiler } from "./investors";
import { displayFilerName, formatDate, formatDealAmount, isSellArticle } from "./format";

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
  code: string
): string {
  const { investorCount, buyCount, sellCount, totalBuyAmount, totalSellAmount, firstDealDate, latestDealDate, topCategory } =
    summary;

  let text = `${stockName}（${code}）は、${formatDate(firstDealDate)}〜${formatDate(latestDealDate)}の間に${
    investorCount > 0 ? `${investorCount}者の投資家から` : ""
  }大量保有・変更報告書が${buyCount + sellCount}件提出されています。`;
  if (buyCount > 0) text += `うち買い方向（取得）が${buyCount}件（推定合計${formatDealAmount(totalBuyAmount)}）`;
  if (sellCount > 0) {
    text += `${buyCount > 0 ? "、" : "うち"}売り方向（譲渡・売却）が${sellCount}件（推定合計${formatDealAmount(totalSellAmount)}）`;
  }
  if (buyCount > 0 || sellCount > 0) text += "です。";
  if (topCategory) text += `提出者の分類では「${topCategory}」による届出が最も多くなっています。`;
  return text;
}

/**
 * 銘柄ページの見出し直下に置く直答文。h1（「◯◯の大株主・株主構成」）＝想定クエリに、
 * ページの1文目でそのまま答えるためのもの（GEO=生成AI検索での引用最適化）。
 * 一覧・表を読まないと分からなかった「誰が何%持っているか」を文章にする。
 * filersは保有比率の降順で渡ってくる前提（getFilersByStockCode）。
 * 保有比率は開示のたびに変わるので、必ず開示日を添える（日付の無い数字を引用させない）。
 */
export function formatStockHolderLead(
  stockName: string,
  code: string,
  filers: StockFiler[]
): string {
  if (filers.length === 0) return "";
  const [top, second] = filers;
  let text =
    `${stockName}（${code}）の株式について大量保有報告書（5%ルール）を提出している投資家は` +
    `${filers.length}者です。`;
  if (top.latestRatio !== null) {
    text +=
      `保有比率が最も高いのは${displayFilerName(top.filerName)}（${top.category}）の` +
      `${top.latestRatio}%` +
      (top.latestDiscDate ? `（${formatDate(top.latestDiscDate)}の開示時点）` : "") +
      "です。";
    if (second && second.latestRatio !== null) {
      text += `次いで${displayFilerName(second.filerName)}が${second.latestRatio}%を保有しています。`;
    }
  }
  return text;
}
