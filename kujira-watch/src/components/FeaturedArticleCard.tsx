import Image from "next/image";
import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { excerptFromHtml, formatDate, formatDealAmount } from "@/lib/format";
import DealTypeBadge from "./DealTypeBadge";

export default function FeaturedArticleCard({ article }: { article: ArticleContent }) {
  return (
    <Link
      href={`/articles/${article.id}`}
      className="group relative mb-10 flex min-h-[22rem] flex-col justify-end overflow-hidden bg-brand-navy text-white"
    >
      {article.eyecatch && (
        <div className="absolute inset-0">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || article.title}
            fill
            priority
            className="object-cover opacity-70 transition-transform duration-300 group-hover:scale-105"
            sizes="100vw"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-brand-navy via-brand-navy/60 to-transparent" />
        </div>
      )}
      <div className="relative p-6 sm:p-10">
        <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-1">
          <span className="kicker text-brand-gold-bright">注目の一件</span>
          <DealTypeBadge dealType={article.dealType} />
          <span className="kicker text-white/60">{formatDate(article.dealDate)}</span>
        </div>
        <h2 className="font-serif text-2xl font-bold leading-snug sm:text-3xl">
          {article.title}
        </h2>
        <p className="mt-3 text-sm text-white/80">
          {article.stockName}（{article.stockCode}） ・ {formatDealAmount(article.dealAmount)}
        </p>
        <p className="mt-3 max-w-2xl text-sm text-white/70">
          {excerptFromHtml(article.body, 90)}
        </p>
      </div>
    </Link>
  );
}
