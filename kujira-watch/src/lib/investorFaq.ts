import type { FaqItem } from "@/lib/faqData";
import type { FilerHolding } from "@/lib/investors";
import { displayFilerName } from "@/lib/format";

// 投資家ページのFAQ（可視ブロック＋FAQPage構造化データの共通ソース）。
// 「◯◯（ファンド名） 保有銘柄」で検索した人の質問にそのページのEDINETデータから答える。
// stockFaq.tsと同じ方針で、構造化データと可視コンテンツを必ずこの戻り値から両方作ること。
export function buildInvestorFaqItems(
  filerName: string,
  majorHoldings: FilerHolding[],
  recentBuys: FilerHolding[]
): FaqItem[] {
  const items: FaqItem[] = [];
  const name = displayFilerName(filerName);

  if (majorHoldings.length > 0) {
    const names = majorHoldings
      .slice(0, 3)
      .map((h) => `${h.issuerName}（${h.issuerCode}）`)
      .join("、");
    const suffix = majorHoldings.length > 3 ? `をはじめ計${majorHoldings.length}銘柄` : "";
    items.push({
      question: `${name}の主な保有銘柄は？`,
      answer: `EDINETの大量保有報告書では、${names}${suffix}の保有を${name}が開示しています。各銘柄の保有比率はこのページの一覧で確認できます。`,
      category: "investor-page",
    });
  }

  if (recentBuys.length > 0) {
    const names = recentBuys
      .slice(0, 3)
      .map((h) => `${h.issuerName}（${h.issuerCode}）`)
      .join("、");
    items.push({
      question: `${name}が直近で買い増した銘柄は？`,
      answer: `直近の開示で保有比率が上昇したのは${names}です（前回開示との比較。詳細な比率の推移は本ページの取引一覧を参照）。`,
      category: "investor-page",
    });
  }

  return items;
}
