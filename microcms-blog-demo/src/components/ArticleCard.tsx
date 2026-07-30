import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { formatDate, formatDealAmount } from "@/lib/format";
import DealTypeBadge from "./DealTypeBadge";

export default function ArticleCard({ article }: { article: ArticleContent }) {
  return (
    <Link
      href={`/articles/${article.id}`}
      className="flex flex-col overflow-hidden rounded-lg bg-white p-4 shadow-sm ring-1 ring-gray-200 transition-shadow hover:shadow-md"
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <DealTypeBadge dealType={article.dealType} />
        <span className="text-xs text-gray-500">{formatDate(article.dealDate)}</span>
      </div>
      <h2 className="text-lg font-semibold text-brand-navy">{article.title}</h2>
      <p className="mt-2 text-sm text-gray-600">
        {article.stockName}（{article.stockCode}） ・{" "}
        {formatDealAmount(article.dealAmount)}
      </p>
    </Link>
  );
}
