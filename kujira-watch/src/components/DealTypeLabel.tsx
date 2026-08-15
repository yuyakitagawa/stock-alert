import type { DealType } from "@/types/article";
import { DEAL_TYPE_COLORS } from "@/lib/dealTypeInfo";

// 一覧で大量に並べる用の軽量な投資家分類ラベル（サーバーコンポーネント）。
// DealTypeBadgeはMUIのChip+Tooltipで"use client"のため、/investorsのように
// 600行以上並ぶページではRSCペイロードとDOMが膨らみ、HTMLが1.5MBに達していた
// （TTFBも約1.9秒）。行の識別に必要なのは色ドットと分類名だけなので、
// ツールチップを持たない素のspanに置き換える。分類の説明は/about#dealtype-glossaryにある。
export default function DealTypeLabel({ dealType }: { dealType: DealType }) {
  if (!dealType) return null;
  const colors = DEAL_TYPE_COLORS[dealType] ?? DEAL_TYPE_COLORS.その他;
  return (
    <span className="kicker inline-flex items-center gap-1 whitespace-nowrap" style={{ color: colors.text }}>
      <span
        aria-hidden
        style={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: colors.dot, display: "inline-block" }}
      />
      {dealType}
    </span>
  );
}
