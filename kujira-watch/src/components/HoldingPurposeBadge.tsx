import { PURPOSE_COLORS, PURPOSE_DESCRIPTIONS, type HoldingPurpose } from "@/lib/disclosures";
import DotBadge from "./DotBadge";

// 大量保有報告書の「保有目的」欄を5区分に寄せたバッジ。
// 同じ「5%取得」でも、純投資と重要提案行為等では読み手にとっての意味がまるで違う。
// 分類はあくまで自由記述からの機械判定なので、原文はバッジの隣に併記する側で出す。
export default function HoldingPurposeBadge({ purpose }: { purpose: HoldingPurpose }) {
  const color = PURPOSE_COLORS[purpose];
  return (
    <DotBadge
      label={purpose}
      dotColor={color}
      color={color}
      tint
      tooltip={PURPOSE_DESCRIPTIONS[purpose]}
    />
  );
}
