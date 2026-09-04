import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import EnArticleCard from "@/components/EnArticleCard";
import { excerptFromHtml, isSellArticle, toDateAttr } from "@/lib/format";
import { getArticleDetail, getArticlesByStockCode, getTranslatedArticleRefs } from "@/lib/microcms";
import { SITE_URL, X_HANDLE } from "@/lib/site";
import {
  EN_SITE_URL,
  SITE_NAME_EN,
  dealTypeLabelEn,
  formatDateEn,
  formatDealAmountOrCorrectionEn,
  isTranslated,
} from "@/lib/en";
import { isIndexableEnArticle, supersededArticleIds } from "@/lib/articleIndexability";

// 日本語版と同じ理由でISRの保持を1日にする（本文は公開後ほぼ変わらない）。
export const revalidate = 86400;

// generateStaticParams が無い動的セグメントはNext 16ではリクエスト毎のSSRになり、
// 何度アクセスしてもCDNキャッシュに乗らない。一部でも事前生成するとルート全体がISR扱いに
// なるので、英訳済み記事の新しい順に少数だけ事前生成する（ビルド時間＝Vercelの課金を抑える）。
const PRERENDERED_EN_ARTICLES = 10;

export async function generateStaticParams() {
  // 取得に失敗しても空配列を返してビルドは通す（日本語版と同じ判断）。
  try {
    const articles = await getTranslatedArticleRefs();
    return articles.slice(0, PRERENDERED_EN_ARTICLES).map((article) => ({ id: article.id }));
  } catch {
    return [];
  }
}

// 日本語版と同じカニバリ判定（同一「銘柄×提出者」で最新1本だけをindex）。判定を揃えないと
// 日本語側がnoindexなのに英語側がindexという食い違いが出る。
async function isSupersededArticle(id: string, stockCode: string | undefined): Promise<boolean> {
  if (!stockCode) return false;
  const { contents } = await getArticlesByStockCode(stockCode);
  return supersededArticleIds(contents).has(id);
}

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const article = await getArticleDetail(id).catch(() => null);
  if (!article || !isTranslated(article)) return {};

  const label = dealTypeLabelEn(article.dealType);
  const description = [
    `${article.stockName} (${article.stockCode})`,
    label ? ` — ${label}` : "",
    `. ${excerptFromHtml(article.bodyEn as string)}`,
  ].join("");
  const url = `${EN_SITE_URL}/articles/${id}`;
  const superseded = await isSupersededArticle(id, article.stockCode);

  return {
    title: { absolute: article.titleEn as string },
    description,
    // 英語版は日本語版より厳しい基準でnoindex（判定は lib/articleIndexability.ts に集約）。
    ...(isIndexableEnArticle(article) && !superseded ? {} : { robots: { index: false, follow: true } }),
    alternates: { canonical: url, languages: { ja: `${SITE_URL}/articles/${id}`, en: url } },
    openGraph: {
      type: "article",
      url,
      title: article.titleEn,
      description,
      publishedTime: article.dealDate,
      modifiedTime: article.dealDate,
      ...(article.eyecatch ? { images: [article.eyecatch.url] } : {}),
    },
    twitter: {
      card: "summary_large_image",
      site: X_HANDLE,
      creator: X_HANDLE,
      title: article.titleEn,
      description,
    },
  };
}

export default async function EnArticleDetailPage({ params }: Props) {
  const { id } = await params;

  const article = await getArticleDetail(id).catch((error: unknown) => {
    if (error instanceof Error && error.message.includes("status: 404")) {
      notFound();
    }
    throw error;
  });

  // 未翻訳の記事は英語版では404扱い（日英混在ページを出さない）
  if (!isTranslated(article)) {
    notFound();
  }
  const titleEn = article.titleEn as string;
  const bodyEn = article.bodyEn as string;

  const label = dealTypeLabelEn(article.dealType);
  const url = `${EN_SITE_URL}/articles/${id}`;

  const refs = await getTranslatedArticleRefs();
  const relatedArticles = refs
    .filter((a) => a.id !== id && (a.stockCode === article.stockCode || a.dealType === article.dealType))
    .slice(0, 4);

  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: titleEn,
    url,
    datePublished: article.dealDate,
    dateModified: article.dealDate,
    inLanguage: "en",
    mainEntityOfPage: { "@type": "WebPage", "@id": url },
    about: {
      "@type": "Corporation",
      name: article.stockName,
      tickerSymbol: article.stockCode,
    },
    articleSection: label,
    author: { "@type": "Organization", name: SITE_NAME_EN, url: EN_SITE_URL },
    publisher: {
      "@type": "Organization",
      name: SITE_NAME_EN,
      url: EN_SITE_URL,
      logo: { "@type": "ImageObject", url: `${SITE_URL}/logo` },
    },
    ...(article.eyecatch ? { image: article.eyecatch.url } : {}),
    ...(article.sourceUrl ? { citation: article.sourceUrl } : {}),
  };

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Latest", item: EN_SITE_URL },
      { "@type": "ListItem", position: 2, name: titleEn, item: url },
    ],
  };

  const facts: { term: string; body: React.ReactNode }[] = [
    {
      term: "Stock",
      body: (
        <a
          href={`${SITE_URL}/stocks/${article.stockCode}`}
          hrefLang="ja"
          className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
        >
          {article.stockName} ({article.stockCode}) — holding history (Japanese)
        </a>
      ),
    },
    {
      term: "Deal date",
      body: <time dateTime={toDateAttr(article.dealDate)}>{formatDateEn(article.dealDate)}</time>,
    },
    { term: "Deal size", body: formatDealAmountOrCorrectionEn(article) },
    ...(article.filerName ? [{ term: "Filer", body: article.filerName }] : []),
    ...(article.sourceUrl
      ? [
          {
            term: "Source",
            body: (
              <a
                href={article.sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-brand-blue hover:underline"
              >
                View original filing
              </a>
            ),
          },
        ]
      : []),
  ];

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
      <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 border-b border-rule px-6 py-3 text-xs text-ink-tertiary">
        <Link href="/" className="flex-none hover:text-brand-blue">Latest</Link>
        <span aria-hidden>/</span>
        <span className="min-w-0 truncate text-ink-secondary">{titleEn}</span>
      </nav>
      {article.eyecatch && (
        <div className="relative aspect-video w-full bg-section-tint">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || titleEn}
            fill
            priority
            className="object-cover"
            sizes="(min-width: 768px) 768px, 100vw"
          />
        </div>
      )}
      <div className="p-6 sm:p-10">
        <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs font-semibold uppercase tracking-wide">
          {label && <span className="text-brand-navy">{label}</span>}
          {isSellArticle(article.tags) && <span className="text-loss">Sell</span>}
        </div>
        <h1 className="mb-4 text-2xl font-bold leading-snug text-brand-navy sm:text-3xl">{titleEn}</h1>
        <dl className="m-0 mb-8 grid grid-cols-1 gap-4 border-y border-rule py-4 sm:grid-cols-2">
          {facts.map((fact) => (
            <div key={fact.term}>
              <dt className="text-xs uppercase tracking-wide text-ink-tertiary">{fact.term}</dt>
              <dd className="m-0 mt-1 font-medium text-ink">{fact.body}</dd>
            </div>
          ))}
        </dl>
        <div
          className="prose max-w-none prose-headings:text-brand-navy prose-a:text-brand-blue"
          dangerouslySetInnerHTML={{ __html: bodyEn }}
        />
        <p className="mt-8 border-t border-rule pt-4 text-xs leading-relaxed text-ink-tertiary">
          Deal size is an estimate (shares outstanding × share price × change in holding ratio); EDINET
          filings do not state a yen amount. This article is generated from the filing&apos;s facts and is
          not investment advice. Read the{" "}
          <a href={`${SITE_URL}/articles/${id}`} hrefLang="ja" className="text-brand-blue hover:underline">
            Japanese edition of this article
          </a>{" "}
          for the latest data.
        </p>
        {relatedArticles.length > 0 && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-5 text-xl font-bold text-brand-navy">Related articles</h2>
            <ul className="m-0 grid list-none grid-cols-1 gap-4 p-0 sm:grid-cols-2">
              {relatedArticles.map((related) => (
                <li key={related.id}>
                  <EnArticleCard article={related} headingLevel="h3" />
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </article>
  );
}
