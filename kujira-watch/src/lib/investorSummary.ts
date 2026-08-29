import type { FilerHolding } from "./investors";
import { displayFilerName, formatDate } from "./format";

// 投資家ページ（/investors/[filer]）の冒頭に置く直答文とファクトボックスの元データ。
// 「〇〇はどんな投資家で、いま何を持っているのか」というAI検索・検索クエリに、
// ページの1文目でそのまま答えるために作る（GEO=生成AI検索での引用最適化）。
// stockSummary.ts と同じ方針で、既にDBにある事実を集計するだけ＝LLM呼び出しをしない。

export type InvestorSummary = {
  disclosureCount: number;
  stockCount: number;
  latestDiscDate: string;
  firstDiscDate: string;
  topHolding: { issuerName: string; issuerCode: string; holdingRatio: number } | null;
  buyCount: number;
  sellCount: number;
};

export function buildInvestorSummary(
  holdings: FilerHolding[],
  majorHoldings: FilerHolding[],
  recentBuys: FilerHolding[],
  recentSells: FilerHolding[]
): InvestorSummary | null {
  if (holdings.length === 0) return null;
  const dates = holdings.map((h) => h.discDate).sort();
  // majorHoldingsは呼び出し側で保有比率の降順に並んでいる（比率不明は末尾）。
  const top = majorHoldings.find((h) => h.holdingRatio !== null);
  return {
    disclosureCount: holdings.length,
    stockCount: majorHoldings.length,
    firstDiscDate: dates[0],
    latestDiscDate: dates[dates.length - 1],
    topHolding: top
      ? { issuerName: top.issuerName, issuerCode: top.issuerCode, holdingRatio: top.holdingRatio! }
      : null,
    buyCount: recentBuys.length,
    sellCount: recentSells.length,
  };
}

/**
 * 直答文（1文目でクエリに答える）。
 * 「◯◯は、EDINETの大量保有報告書でN銘柄・M件の保有を開示している【分類】です。」から始め、
 * 最新の開示日と最大保有銘柄まで1段落に収める。日付を必ず含めるのは、保有比率が開示のたびに
 * 変わるため（日付の無い数字を引用されると誤りになる）。
 */
export function formatInvestorSummary(
  filerName: string,
  category: string,
  summary: InvestorSummary
): string {
  const name = displayFilerName(filerName);
  const { disclosureCount, stockCount, latestDiscDate, topHolding, buyCount, sellCount } = summary;

  let text =
    `${name}は、EDINETの大量保有報告書（5%ルール）で${stockCount}銘柄・` +
    `合計${disclosureCount}件の保有を開示している投資家です（分類: ${category}）。`;
  text += `直近の開示は${formatDate(latestDiscDate)}です。`;
  if (topHolding) {
    text +=
      `保有比率が最も高いのは${topHolding.issuerName}（${topHolding.issuerCode}）の` +
      `${topHolding.holdingRatio}%です。`;
  }
  if (buyCount > 0 || sellCount > 0) {
    const parts: string[] = [];
    if (buyCount > 0) parts.push(`買い増しが${buyCount}銘柄`);
    if (sellCount > 0) parts.push(`売却が${sellCount}銘柄`);
    // 新規保有の開示は前回比率が無く買い増し・売却のどちらにも入らないため、
    // 「全銘柄の内訳」と読めない言い回しにする（実例: 最新の開示が全部新規のファンドで
    // 売却だけが数えられ、買っていないように読めてしまう）。
    text += `各銘柄の最新の開示のうち、前回の保有比率と比較できるものでは${parts.join("、")}です。`;
  }
  return text;
}
