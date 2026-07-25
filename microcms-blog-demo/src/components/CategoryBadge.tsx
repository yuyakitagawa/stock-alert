import Link from "next/link";
import type { Category } from "@/types/article";

export default function CategoryBadge({ category }: { category: Category }) {
  return (
    <Link
      href={`/category/${encodeURIComponent(category)}`}
      className="inline-flex items-center rounded-full border border-gray-300 px-2.5 py-0.5 text-xs font-medium text-gray-600 hover:bg-gray-50"
    >
      {category}
    </Link>
  );
}
