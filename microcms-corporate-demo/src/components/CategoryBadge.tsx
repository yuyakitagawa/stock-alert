import type { NewsCategory } from "@/types/news";

const STYLE: Record<NewsCategory, string> = {
  お知らせ: "border-brand-navy/30 text-brand-navy",
  プレスリリース: "border-brand-blue/40 text-brand-blue",
  IR情報: "border-brand-gold/60 text-amber-700",
  サステナビリティ: "border-brand-green/50 text-emerald-700",
};

export default function CategoryBadge({ category }: { category: NewsCategory }) {
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap rounded-full border bg-white px-2.5 py-0.5 text-xs font-semibold ${STYLE[category]}`}
    >
      {category}
    </span>
  );
}
