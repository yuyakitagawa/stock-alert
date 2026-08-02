import Image from "next/image";
import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { formatDate, formatDealAmount } from "@/lib/format";
import DealTypeBadge from "./DealTypeBadge";

export default function ArticleCard({ article }: { article: ArticleContent }) {
  return (
    <Link
      href={`/articles/${article.id}`}
      className="flex flex-col overflow-hidden rounded-xl bg-white shadow-sm ring-1 ring-gray-200 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg"
    >
      {article.eyecatch && (
        <div className="relative aspect-video w-full bg-section-tint">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || article.title}
            fill
            className="object-cover"
            sizes="(min-width: 640px) 50vw, 100vw"
          />
        </div>
      )}
      <div className="flex flex-1 flex-col p-5">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <DealTypeBadge dealType={article.dealType} />
          <span className="text-xs text-gray-500">{formatDate(article.dealDate)}</span>
        </div>
        <h2 className="text-lg font-semibold leading-snug text-brand-navy">{article.title}</h2>
        <p className="mt-2 text-sm text-gray-600">
          {article.stockName}（{article.stockCode}） ・{" "}
          {formatDealAmount(article.dealAmount)}
        </p>
      </div>
    </Link>
  );
}
