import Link from "next/link";

export default function CategoryBadge({ category }: { category: string | undefined }) {
  if (!category) return null;
  return (
    <Link
      href={`/category/${encodeURIComponent(category)}`}
      className="inline-flex items-center rounded-full border border-brand-navy/30 bg-white px-2.5 py-0.5 text-xs font-semibold text-brand-navy hover:bg-section-tint"
    >
      {category}
    </Link>
  );
}
