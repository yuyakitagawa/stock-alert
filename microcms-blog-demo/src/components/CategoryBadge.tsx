import Link from "next/link";
import type { Category } from "@/types/article";

export default function CategoryBadge({ category }: { category: Category }) {
  return (
    <Link
      href={`/category/${encodeURIComponent(category)}`}
      className="inline-flex items-center rounded-full border border-brand-navy/30 bg-white px-2.5 py-0.5 text-xs font-semibold text-brand-navy hover:bg-section-tint"
    >
      {category}
    </Link>
  );
}
