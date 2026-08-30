import type { DealType } from "@/types/article";
import { DEAL_TYPE_COLORS } from "@/lib/dealTypeInfo";
import DotBadge from "./DotBadge";

export default function CategoryBadge({ dealType }: { dealType: DealType | undefined }) {
  if (!dealType) return null;
  // 一覧のドット版(DealTypeBadge)と同じ分類色ドットを付ける。同じ「分類」を指す
  // バッジなのに、一覧=色ドット・記事詳細=枠付きChipで別物に見えていた。
  // 枠とリンクは記事詳細でカテゴリ一覧への導線であることを示すため残す。
  const dotColor = (DEAL_TYPE_COLORS[dealType] ?? DEAL_TYPE_COLORS.その他).dot;
  return (
    <DotBadge
      label={dealType}
      dotColor={dotColor}
      color="var(--brand-navy)"
      bordered
      href={`/category/${encodeURIComponent(dealType)}`}
      sx={{ "&:hover": { color: "brand.gold", borderColor: "brand.gold" } }}
    />
  );
}
