import Image from "next/image";
import Link from "next/link";
import type { ArticleContent } from "@/types/article";
import { formatDate, formatDealAmount } from "@/lib/format";
import type { Locale } from "@/lib/i18n";
import DealDirectionBadge from "./DealDirectionBadge";
import DealTypeBadge from "./DealTypeBadge";

export default function ArticleCard({
  article,
  locale = "ja",
}: {
  article: ArticleContent;
  locale?: Locale;
}) {
  const href = locale === "en" ? `/en/articles/${article.id}` : `/articles/${article.id}`;
  const title = locale === "en" ? article.titleEn ?? article.title : article.title;
  return (
    <Link
      href={href}
      className="group flex flex-col overflow-hidden border-t border-rule pt-4 transition-opacity hover:opacity-80"
    >
      {article.eyecatch && (
        <div className="relative mb-4 aspect-video w-full bg-section-tint">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || title}
            fill
            className="object-cover"
            sizes="(min-width: 640px) 50vw, 100vw"
          />
        </div>
      )}
      <div className="flex flex-1 flex-col">
        <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
          <DealTypeBadge dealType={article.dealType} locale={locale} />
          <DealDirectionBadge tags={article.tags} locale={locale} />
          <span className="kicker text-brand-navy/50">{formatDate(article.dealDate, locale)}</span>
        </div>
        <h2 className="text-lg font-bold leading-snug text-brand-navy underline decoration-brand-gold/0 decoration-2 underline-offset-4 group-hover:decoration-brand-gold/70">
          {title}
        </h2>
        <p className="mt-2 text-sm text-foreground/60">
          {locale === "en"
            ? `${article.stockName} (${article.stockCode}) · ${formatDealAmount(article.dealAmount, locale)}`
            : `${article.stockName}（${article.stockCode}） ・ ${formatDealAmount(article.dealAmount, locale)}`}
        </p>
      </div>
    </Link>
  );
}
