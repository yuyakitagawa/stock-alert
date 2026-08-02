import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { excerptFromHtml, formatDate, formatDealAmount } from "@/lib/format";
import DealTypeBadge from "./DealTypeBadge";

export default function FeaturedArticleCard({ article }: { article: ArticleContent }) {
  return (
    <Link
      href={`/articles/${article.id}`}
      className="mb-6 flex flex-col overflow-hidden rounded-xl bg-brand-navy p-6 text-white shadow-md transition-transform hover:-translate-y-0.5 hover:shadow-lg sm:p-8"
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-brand-gold px-2.5 py-0.5 text-xs font-bold text-brand-navy">
          注目の一件
        </span>
        <DealTypeBadge dealType={article.dealType} />
        <span className="text-xs text-white/70">{formatDate(article.dealDate)}</span>
      </div>
      <h2 className="text-2xl font-bold leading-snug">{article.title}</h2>
      <p className="mt-2 text-sm text-white/80">
        {article.stockName}（{article.stockCode}） ・ {formatDealAmount(article.dealAmount)}
      </p>
      <p className="mt-3 text-sm text-white/70">{excerptFromHtml(article.body, 90)}</p>
    </Link>
  );
}
