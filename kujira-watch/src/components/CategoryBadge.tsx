import Link from "next/link";

export default function CategoryBadge({ category }: { category: string | undefined }) {
  if (!category) return null;
  return (
    <Link
      href={`/category/${encodeURIComponent(category)}`}
      className="kicker inline-flex items-center border-b border-brand-navy/40 text-brand-navy transition-colors hover:border-brand-gold hover:text-brand-gold"
    >
      {category}
    </Link>
  );
}
