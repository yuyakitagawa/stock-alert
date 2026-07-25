import Image from "next/image";
import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { formatDate, formatDealAmount } from "@/lib/format";
import DealTypeBadge from "./DealTypeBadge";

export default function ArticleCard({ article }: { article: ArticleContent }) {
  return (
    <Link
      href={`/articles/${article.id}`}
      className="flex flex-col overflow-hidden rounded-lg border border-gray-200 transition-shadow hover:shadow-md sm:flex-row"
    >
      <div className="relative h-48 w-full flex-shrink-0 bg-gray-100 sm:h-auto sm:w-56">
        {article.eyecatch ? (
          <Image
            src={article.eyecatch.url}
            alt={article.title}
            fill
            sizes="(min-width: 640px) 224px, 100vw"
            className="object-cover"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-gray-400">
            No Image
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-2 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <DealTypeBadge dealType={article.dealType} />
          <span className="text-xs text-gray-500">
            {formatDate(article.dealDate)}
          </span>
        </div>
        <h2 className="text-lg font-semibold text-gray-900">{article.title}</h2>
        <p className="text-sm text-gray-600">
          {article.stockName}（{article.stockCode}） ・{" "}
          {formatDealAmount(article.dealAmount)}
        </p>
      </div>
    </Link>
  );
}
