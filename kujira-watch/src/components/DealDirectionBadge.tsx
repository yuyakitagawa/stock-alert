import { isSellArticle } from "@/lib/format";
import { UI, type Locale } from "@/lib/i18n";

// 買い方向の記事にはバッジを出さない（従来通りの見た目を維持し、売り方向のみ目立たせる）。
export default function DealDirectionBadge({ tags, locale = "ja" }: { tags?: string; locale?: Locale }) {
  if (!isSellArticle(tags)) return null;
  const t = UI[locale];
  return (
    <span
      title={t.sellBadgeTitle}
      className="kicker inline-flex items-center gap-1.5 text-rose-700"
    >
      <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-rose-600" />
      {t.sellBadge}
    </span>
  );
}
