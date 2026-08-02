import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import CategoryBadge from "@/components/CategoryBadge";
import DealTypeBadge from "@/components/DealTypeBadge";
import { excerptFromHtml, formatDate, formatDealAmount } from "@/lib/format";
import { getArticleDetail } from "@/lib/microcms";
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

  return {
    title: article.title,
    description,
    alternates: { canonical: url },
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

  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: article.title,
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
    publisher: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
    ...(article.sourceUrl ? { citation: article.sourceUrl } : {}),
  };

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      ...(category
        ? [
            {
              "@type": "ListItem",
              position: 2,
              name: category,
              item: `${SITE_URL}/category/${encodeURIComponent(category)}`,
            },
          ]
        : []),
      { "@type": "ListItem", position: category ? 3 : 2, name: article.title, item: url },
    ],
  };

  return (
    <article className="overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-gray-200">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="border-b border-gray-100 px-6 py-3 text-xs text-gray-500">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {category && (
          <>
            {" / "}
            <Link href={`/category/${encodeURIComponent(category)}`} className="hover:text-brand-blue">
              {category}
            </Link>
          </>
        )}
        {" / "}
        <span className="text-gray-700">{article.title}</span>
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
      <div className="p-6">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <DealTypeBadge dealType={article.dealType} />
          <CategoryBadge category={category} />
        </div>
        <h1 className="mb-4 text-3xl font-bold text-brand-navy">{article.title}</h1>
        <dl className="mb-6 grid grid-cols-2 gap-x-4 gap-y-3 rounded-lg bg-section-tint p-4 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-gray-500">銘柄</dt>
            <dd className="font-medium text-brand-navy">
              {article.stockName}（{article.stockCode}）
            </dd>
          </div>
          <div>
            <dt className="text-gray-500">取引日</dt>
            <dd className="font-medium text-brand-navy">{formatDate(article.dealDate)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">金額規模</dt>
            <dd className="font-medium text-brand-navy">
              {formatDealAmount(article.dealAmount)}
            </dd>
          </div>
          {article.sourceUrl && (
            <div>
              <dt className="text-gray-500">出典</dt>
              <dd>
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
          className="prose max-w-none prose-a:text-brand-blue"
          dangerouslySetInnerHTML={{ __html: article.body }}
        />
        {tags && tags.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-2 border-t border-gray-200 pt-4">
            {tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-gray-300 px-2.5 py-0.5 text-xs text-gray-600"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}
