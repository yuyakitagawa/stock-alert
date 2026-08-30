import { isSellArticle } from "@/lib/format";
import { UI } from "@/lib/i18n";
import DotBadge from "./DotBadge";

// 買い方向の記事にはバッジを出さない（従来通りの見た目を維持し、売り方向のみ目立たせる）。
export default function DealDirectionBadge({
  tags,
  onDark = false,
}: {
  tags?: string;
  // ダーク地ではerror.mainの文字が沈むため、文字は白・売りの赤はドットのみで示す。
  onDark?: boolean;
}) {
  if (!isSellArticle(tags)) return null;
  return (
    <DotBadge
      label={UI.sellBadge}
      dotColor="var(--loss)"
      color={onDark ? "var(--ink-on-dark)" : "var(--loss)"}
      tooltip={UI.sellBadgeTitle}
    />
  );
}
