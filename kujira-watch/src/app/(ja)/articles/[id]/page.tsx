import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import CategoryBadge from "@/components/CategoryBadge";
import DealDirectionBadge from "@/components/DealDirectionBadge";
import DealTypeBadge from "@/components/DealTypeBadge";
import ArticleCard from "@/components/ArticleCard";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { excerptFromHtml, formatDate, formatDealAmount } from "@/lib/format";
import { getArticleDetail, getArticleList } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { categoryLabel } from "@/types/article";

// Route segment config requires a literal value (cannot import from lib/microcms).
export const revalidate = 60;

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const article = await getArticleDetail(id).catch(() => null);
  if (!article) return {};

  const description = `${article.stockName}(${article.stockCode})の${article.dealType}を解説。${excerptFromHtml(article.body)}`;
  const url = `${SITE_URL}/articles/${id}`;
  const hasEn = Boolean(article.titleEn && article.bodyEn);

  return {
    title: article.title,
    description,
    alternates: {
      canonical: url,
      ...(hasEn ? { languages: { ja: url, en: `${SITE_URL}/en/articles/${id}` } } : {}),
    },
    openGraph: {
      type: "article",
      url,
      title: article.title,
      description,
      publishedTime: article.publishedAt,
      modifiedTime: article.updatedAt,
      ...(article.eyecatch ? { images: [article.eyecatch.url] } : {}),
    },
    twitter: {
      card: "summary_large_image",
      title: article.title,
      description,
    },
  };
}

export default async function ArticleDetailPage({ params }: Props) {
  const { id } = await params;

  const article = await getArticleDetail(id).catch((error: unknown) => {
    if (error instanceof Error && error.message.includes("status: 404")) {
      notFound();
    }
    throw error;
  });

  const tags = article.tags
    ?.split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);

  const category = article.dealType ? categoryLabel(article.dealType) : undefined;
  const url = `${SITE_URL}/articles/${id}`;
  const dealDateOnly = article.dealDate.slice(0, 10);

  const { contents: sameCategoryArticles } = article.dealType
    ? await getArticleList({ dealType: article.dealType, limit: 5 })
    : { contents: [] };
  const relatedArticles = sameCategoryArticles.filter((a) => a.id !== id).slice(0, 4);

  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: article.title,
    url,
    datePublished: article.publishedAt ?? article.createdAt,
    dateModified: article.updatedAt,
    inLanguage: "ja",
    mainEntityOfPage: { "@type": "WebPage", "@id": url },
    about: {
      "@type": "Corporation",
      name: article.stockName,
      tickerSymbol: article.stockCode,
    },
    articleSection: category,
    // 記事は人手ではなくAIがEDINET開示等の事実情報から生成しているため、
    // authorはサイト運営組織そのもの（Organization）とする。
    author: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
    publisher: {
      "@type": "Organization",
      name: SITE_NAME,
      url: SITE_URL,
      logo: { "@type": "ImageObject", url: `${SITE_URL}/logo` },
    },
    ...(article.eyecatch ? { image: article.eyecatch.url } : {}),
    ...(article.sourceUrl ? { citation: article.sourceUrl } : {}),
  };

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      {
        "@type": "ListItem",
        position: 2,
        name: formatDate(article.dealDate),
        item: `${SITE_URL}/date/${dealDateOnly}`,
      },
      { "@type": "ListItem", position: 3, name: article.title, item: url },
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
      <nav aria-label="パンくずリスト" className="border-b border-rule px-6 py-3 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href={`/date/${dealDateOnly}`} className="hover:text-brand-blue">
          {formatDate(article.dealDate)}
        </Link>
        {" / "}
        <span className="text-foreground/70">{article.title}</span>
      </nav>
      {article.eyecatch && (
        <div className="relative aspect-video w-full bg-section-tint">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || article.title}
            fill
            priority
            className="object-cover"
            sizes="(min-width: 768px) 768px, 100vw"
          />
        </div>
      )}
      <div className="p-6 sm:p-10">
        <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <DealTypeBadge dealType={article.dealType} />
          <DealDirectionBadge tags={article.tags} />
          <CategoryBadge dealType={article.dealType} />
        </div>
        <h1 className="mb-4 text-2xl font-bold leading-snug text-brand-navy sm:text-3xl">
          {article.title}
        </h1>
        {article.dealType && (
          <p className="mb-6 text-xs text-foreground/50">
            {DEAL_TYPE_DESCRIPTIONS[article.dealType]}{" "}
            <Link href="/about#dealtype-glossary" className="text-brand-blue hover:underline">
              分類の一覧を見る
            </Link>
          </p>
        )}
        <dl className="mb-8 grid grid-cols-2 gap-x-4 gap-y-4 border-y border-rule py-4 text-sm sm:grid-cols-4">
          <div>
            <dt className="kicker text-foreground/40">銘柄</dt>
            <dd className="mt-1 font-medium">
              <Link
                href={`/stocks/${article.stockCode}`}
                className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
              >
                {article.stockName}（{article.stockCode}）の履歴を見る
              </Link>
            </dd>
          </div>
          <div>
            <dt className="kicker text-foreground/40">取引日</dt>
            <dd className="mt-1 font-medium text-brand-navy">{formatDate(article.dealDate)}</dd>
          </div>
          <div>
            <dt className="kicker text-foreground/40">金額規模</dt>
            <dd className="mt-1 font-medium text-brand-navy">
              {formatDealAmount(article.dealAmount)}
            </dd>
          </div>
          {article.sourceUrl && (
            <div>
              <dt className="kicker text-foreground/40">出典</dt>
              <dd className="mt-1">
                <a
                  href={article.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-brand-blue hover:underline"
                >
                  元記事を見る
                </a>
              </dd>
            </div>
          )}
        </dl>
        <div
          className="prose max-w-none prose-headings:text-brand-navy prose-a:text-brand-blue first:prose-p:first-letter:float-left first:prose-p:first-letter:mr-2 first:prose-p:first-letter:text-5xl first:prose-p:first-letter:font-bold first:prose-p:first-letter:text-brand-navy"
          dangerouslySetInnerHTML={{ __html: article.body }}
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
              関連記事（{category}）
            </h2>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              {relatedArticles.map((related) => (
                <ArticleCard key={related.id} article={related} />
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}
