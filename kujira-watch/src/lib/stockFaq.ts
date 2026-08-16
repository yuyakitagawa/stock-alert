import type { FaqItem } from "@/lib/faqData";
import type { StockHoldingRow } from "@/lib/investors";
import { disclosureDocLabel } from "@/lib/disclosures";
import { formatDate } from "@/lib/format";

// 銘柄ページのFAQ（可視ブロック＋FAQPage構造化データの共通ソース）。
// 「◯◯ 大量保有」「◯◯ 大株主」で検索した人の質問に、そのページのEDINETデータから
// 直接答える。構造化データと可視コンテンツはGoogleのガイドラインで一致が必須のため、
// 両方をこの1関数の戻り値から作ること（faqData.tsxと同じ方針）。
export function buildStockFaqItems(
  stockName: string,
  code: string,
  filers: { filerName: string }[],
  holdings: StockHoldingRow[]
): FaqItem[] {
  const items: FaqItem[] = [];

  if (filers.length > 0) {
    const names = filers.slice(0, 3).map((f) => f.filerName).join("、");
    const suffix = filers.length > 3 ? `をはじめ計${filers.length}者` : "";
    items.push({
      question: `${stockName}（${code}）の株を大量保有している投資家は？`,
      answer: `EDINETの大量保有報告書（5%ルール）では、${names}${suffix}が${stockName}（${code}）の株式保有を開示しています。各投資家の保有比率の推移はこのページの一覧で確認できます。`,
      category: "stock-page",
    });
  }

  const latest = holdings.find((h) => h.holdingRatio !== null);
  if (latest) {
    items.push({
      question: `${stockName}の直近の大量保有報告書の内容は？`,
      answer: `${formatDate(latest.discDate)}に${latest.filerName}が${disclosureDocLabel(latest)}を提出し、保有比率${latest.holdingRatio}%を開示しています。`,
      category: "stock-page",
    });
  }

  items.push({
    question: "大量保有報告書（5%ルール）とは？",
    answer:
      "上場企業の株式を発行済株式総数の5%を超えて保有した投資家に、金融庁の開示システムEDINETへの提出が義務付けられている書類です。その後保有比率が1%以上変動した場合も変更報告書の提出が必要で、機関投資家やアクティビストなど大口投資家の売買動向を知る一次情報になります。",
    category: "stock-page",
  });

  return items;
}
