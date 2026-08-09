import Image from "next/image";
import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { excerptFromHtml, formatDate, formatDealAmount } from "@/lib/format";
import { UI, type Locale } from "@/lib/i18n";
import DealDirectionBadge from "./DealDirectionBadge";
import DealTypeBadge from "./DealTypeBadge";

export default function FeaturedArticleCard({
  article,
  rank,
  locale = "ja",
}: {
  article: ArticleContent;
  rank: number;
  locale?: Locale;
}) {
  const t = UI[locale];
  const hasImage = Boolean(article.eyecatch);
  const href = locale === "en" ? `/en/articles/${article.id}` : `/articles/${article.id}`;
  const title = locale === "en" ? article.titleEn ?? article.title : article.title;
  const body = locale === "en" ? article.bodyEn ?? article.body : article.body;
  return (
    <Link
      href={href}
      className={`group relative mb-10 flex flex-col overflow-hidden bg-brand-navy text-white ${
        hasImage ? "min-h-[22rem] justify-end" : ""
      }`}
    >
      {article.eyecatch && (
        <div className="absolute inset-0">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || title}
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
          <span className="kicker text-brand-gold-bright">{t.featuredRankLabels[rank] ?? t.featuredFallback}</span>
          <DealTypeBadge dealType={article.dealType} locale={locale} />
          <DealDirectionBadge tags={article.tags} locale={locale} />
          <span className="kicker text-white/60">{formatDate(article.dealDate, locale)}</span>
        </div>
        <h2 className="text-2xl font-bold leading-snug sm:text-3xl">
          {title}
        </h2>
        <p className="mt-3 text-sm text-white/80">
          {locale === "en"
            ? `${article.stockName} (${article.stockCode}) · ${formatDealAmount(article.dealAmount, locale)}`
            : `${article.stockName}（${article.stockCode}） ・ ${formatDealAmount(article.dealAmount, locale)}`}
        </p>
        <p className="mt-3 max-w-2xl text-sm text-white/70">
          {excerptFromHtml(body, 90)}
        </p>
      </div>
    </Link>
  );
}
