import Image from "next/image";
import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { formatDate, formatDealAmount } from "@/lib/format";

export default function ArticleCard({ article }: { article: ArticleContent }) {
  return (
    <Link
      href={`/articles/${article.id}`}
      className="flex flex-col overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-gray-200 transition-shadow hover:shadow-md"
    >
      <div className="relative aspect-[16/9] w-full flex-shrink-0 bg-gray-100">
        {article.eyecatch ? (
          <Image
            src={article.eyecatch.url}
            alt={article.title}
            fill
            sizes="(min-width: 640px) 672px, 100vw"
            className="object-cover"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-gray-400">
            No Image
          </div>
        )}
        <span className="absolute left-3 top-3 inline-flex items-center rounded-full border border-brand-blue bg-white/95 px-2.5 py-0.5 text-xs font-semibold text-brand-blue">
          {article.dealType}
        </span>
      </div>
      <div className="flex flex-1 flex-col gap-2 p-4">
        <span className="text-xs text-gray-500">{formatDate(article.dealDate)}</span>
        <h2 className="text-lg font-semibold text-brand-navy">{article.title}</h2>
        <p className="text-sm text-gray-600">
          {article.stockName}（{article.stockCode}） ・{" "}
          {formatDealAmount(article.dealAmount)}
        </p>
      </div>
    </Link>
  );
}
