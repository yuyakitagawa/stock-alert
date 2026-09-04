import Link from "next/link";
import type { TranslatedArticleRef } from "@/lib/microcms";
import { toDateAttr, isSellArticle } from "@/lib/format";
import { dealTypeLabelEn, formatDateEn, formatDealAmountOrCorrectionEn } from "@/lib/en";

// 英語版（en.kujira-watch.com）の記事カード。日本語側の ArticleCard は文言が日本語固定で
// MUI（client component）に依存するため、英語版はサーバーコンポーネントで軽く出す。
export default function EnArticleCard({
  article,
  headingLevel = "h3",
}: {
  article: TranslatedArticleRef;
  headingLevel?: "h2" | "h3";
}) {
  const Heading = headingLevel;
  const label = dealTypeLabelEn(article.dealType);
  return (
    <article className="rounded-md border border-rule bg-paper p-4 transition-colors hover:border-brand-blue">
      <div className="mb-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-tertiary">
        {label && <span className="font-semibold uppercase tracking-wide text-brand-navy">{label}</span>}
        {isSellArticle(article.tags) && (
          <span className="font-semibold uppercase tracking-wide text-loss" title="Holding ratio decreased">Sell</span>
        )}
        <time dateTime={toDateAttr(article.dealDate)}>{formatDateEn(article.dealDate)}</time>
      </div>
      <Heading className="m-0 text-base font-bold leading-snug">
        <Link href={`/articles/${article.id}`} className="text-brand-blue hover:underline">
          {article.titleEn}
        </Link>
      </Heading>
      <p className="mb-0 mt-1 text-sm text-ink-secondary">
        {article.stockName} ({article.stockCode}) · {formatDealAmountOrCorrectionEn(article)}
      </p>
    </article>
  );
}
