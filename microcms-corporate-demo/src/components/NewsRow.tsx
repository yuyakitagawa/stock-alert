import Link from "next/link";
import type { NewsContent } from "@/types/news";
import { formatDate } from "@/lib/format";
import CategoryBadge from "@/components/CategoryBadge";

export default function NewsRow({ news }: { news: NewsContent }) {
  return (
    <Link
      href={`/news/${news.id}`}
      className="flex flex-col gap-2 border-b border-gray-200 py-4 transition-colors hover:bg-section-tint sm:flex-row sm:items-center sm:gap-6 sm:px-3"
    >
      <span className="text-sm tabular-nums text-gray-500 sm:w-24 sm:flex-shrink-0">
        {formatDate(news.publishedAt ?? news.createdAt)}
      </span>
      <span className="sm:w-28 sm:flex-shrink-0">
        <CategoryBadge category={news.category} />
      </span>
      <span className="text-sm font-medium text-brand-navy sm:flex-1">{news.title}</span>
    </Link>
  );
}
