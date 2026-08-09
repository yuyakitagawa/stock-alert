import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import CategoryBadge from "@/components/CategoryBadge";
import DealDirectionBadge from "@/components/DealDirectionBadge";
import DealTypeBadge from "@/components/DealTypeBadge";
import ArticleCard from "@/components/ArticleCard";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import { excerptFromHtml, formatDate, formatDealAmount } from "@/lib/format";
import { getArticleDetail, getArticleList } from "@/lib/microcms";
import { SITE_NAME_EN, SITE_URL } from "@/lib/site";
import { UI } from "@/lib/i18n";

export const revalidate = 60;

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const article = await getArticleDetail(id).catch(() => null);
  if (!article || !article.titleEn || !article.bodyEn) return {};

  const description = `${article.stockName}(${article.stockCode}) — ${DEAL_TYPE_EN[article.dealType].label}. ${excerptFromHtml(article.bodyEn)}`;
  const url = `${SITE_URL}/en/articles/${id}`;

  return {
    title: article.titleEn,
    description,
    alternates: { canonical: url, languages: { ja: `${SITE_URL}/articles/${id}`, en: url } },
    openGraph: {
      type: "article",
      url,
      title: article.titleEn,
      description,
      publishedTime: article.publishedAt,
      modifiedTime: article.updatedAt,
      ...(article.eyecatch ? { images: [article.eyecatch.url] } : {}),
    },
    twitter: {
      card: "summary_large_image",
      title: article.titleEn,
      description,
    },
  };
}

export default async function EnArticleDetailPage({ params }: Props) {
  const { id } = await params;
  const t = UI.en;

  const article = await getArticleDetail(id).catch((error: unknown) => {
    if (error instanceof Error && error.message.includes("status: 404")) {
      notFound();
    }
    throw error;
  });

  // 未翻訳の記事はEN側では404扱い（日英混在ページを出さない）
  if (!article.titleEn || !article.bodyEn) {
    notFound();
  }

  const tags = article.tags
    ?.split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);

  const dealTypeInfo = DEAL_TYPE_EN[article.dealType];
  const url = `${SITE_URL}/en/articles/${id}`;

  const { contents: sameCategoryArticles } = article.dealType
    ? await getArticleList({ dealType: article.dealType, limit: 5, translatedOnly: true })
    : { contents: [] };
  const relatedArticles = sameCategoryArticles.filter((a) => a.id !== id).slice(0, 4);

  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: article.titleEn,
    url,
    datePublished: article.publishedAt ?? article.createdAt,
    dateModified: article.updatedAt,
    inLanguage: "en",
    mainEntityOfPage: { "@type": "WebPage", "@id": url },
    about: {
      "@type": "Corporation",
      name: article.stockName,
      tickerSymbol: article.stockCode,
    },
    articleSection: dealTypeInfo?.label,
    author: { "@type": "Organization", name: SITE_NAME_EN, url: `${SITE_URL}/en` },
    publisher: {
      "@type": "Organization",
      name: SITE_NAME_EN,
      url: `${SITE_URL}/en`,
      logo: { "@type": "ImageObject", url: `${SITE_URL}/logo` },
    },
    ...(article.eyecatch ? { image: article.eyecatch.url } : {}),
    ...(article.sourceUrl ? { citation: article.sourceUrl } : {}),
  };

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: t.top, item: `${SITE_URL}/en` },
      { "@type": "ListItem", position: 2, name: article.titleEn, item: url },
    ],
  };

  return (
    <article className="overflow-hidden bg-paper">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <nav aria-label={t.breadcrumbAria} className="border-b border-rule px-6 py-3 text-xs text-foreground/50">
        <Link href="/en" className="hover:text-brand-blue">{t.top}</Link>
        {" / "}
        <span className="text-foreground/70">{article.titleEn}</span>
      </nav>
      {article.eyecatch && (
        <div className="relative aspect-video w-full bg-section-tint">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || article.titleEn}
            fill
            priority
            className="object-cover"
            sizes="(min-width: 768px) 768px, 100vw"
          />
        </div>
      )}
      <div className="p-6 sm:p-10">
        <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <DealTypeBadge dealType={article.dealType} locale="en" />
          <DealDirectionBadge tags={article.tags} locale="en" />
          <CategoryBadge dealType={article.dealType} locale="en" />
        </div>
        <h1 className="mb-4 text-2xl font-bold leading-snug text-brand-navy sm:text-3xl">
          {article.titleEn}
        </h1>
        {article.dealType && (
          <p className="mb-6 text-xs text-foreground/50">
            {dealTypeInfo?.description}{" "}
            <Link href="/en/about#dealtype-glossary" className="text-brand-blue hover:underline">
              {t.viewClassificationList}
            </Link>
          </p>
        )}
        <dl className="mb-8 grid grid-cols-2 gap-x-4 gap-y-4 border-y border-rule py-4 text-sm sm:grid-cols-4">
          <div>
            <dt className="kicker text-foreground/40">{t.stockLabel}</dt>
            <dd className="mt-1 font-medium">
              <Link
                href={`/en/stocks/${article.stockCode}`}
                className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
              >
                {article.stockName} ({article.stockCode}){t.viewHistorySuffix}
              </Link>
            </dd>
          </div>
          <div>
            <dt className="kicker text-foreground/40">{t.dealDateLabel}</dt>
            <dd className="mt-1 font-medium text-brand-navy">{formatDate(article.dealDate, "en")}</dd>
          </div>
          <div>
            <dt className="kicker text-foreground/40">{t.dealSizeLabel}</dt>
            <dd className="mt-1 font-medium text-brand-navy">
              {formatDealAmount(article.dealAmount, "en")}
            </dd>
          </div>
          {article.sourceUrl && (
            <div>
              <dt className="kicker text-foreground/40">{t.sourceLabel}</dt>
              <dd className="mt-1">
                <a
                  href={article.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-brand-blue hover:underline"
                >
                  {t.viewOriginal}
                </a>
              </dd>
            </div>
          )}
        </dl>
        <div
          className="prose max-w-none prose-headings:text-brand-navy prose-a:text-brand-blue first:prose-p:first-letter:float-left first:prose-p:first-letter:mr-2 first:prose-p:first-letter:text-5xl first:prose-p:first-letter:font-bold first:prose-p:first-letter:text-brand-navy"
          dangerouslySetInnerHTML={{ __html: article.bodyEn }}
        />
        {tags && tags.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-x-3 gap-y-1 border-t border-rule pt-4 text-xs text-foreground/50">
            {tags.map((tag) => (
              <span key={tag}>#{tag}</span>
            ))}
          </div>
        )}
        {relatedArticles.length > 0 && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-5 text-lg font-bold text-brand-navy">
              {t.relatedArticlesHeading(dealTypeInfo?.label)}
            </h2>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              {relatedArticles.map((related) => (
                <ArticleCard key={related.id} article={related} locale="en" />
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}
