import type { DealType } from "@/types/article";
import { DEAL_TYPE_COLORS, DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import DotBadge from "./DotBadge";

export default function DealTypeBadge({
  dealType,
  onDark = false,
}: {
  dealType: DealType;
  // 注目カードなどダーク地に載せる場合。DEAL_TYPE_COLORS.textはライト背景用の濃色で
  // ダーク地ではコントラストが出ないため、文字は白・分類色はドットのみで示す。
  onDark?: boolean;
}) {
  if (!dealType) return null;
  const colors = DEAL_TYPE_COLORS[dealType] ?? DEAL_TYPE_COLORS.その他;

  return (
    <DotBadge
      label={dealType}
      dotColor={colors.dot}
      color={onDark ? "var(--ink-on-dark)" : colors.text}
      tooltip={DEAL_TYPE_DESCRIPTIONS[dealType]}
    />
  );
}
