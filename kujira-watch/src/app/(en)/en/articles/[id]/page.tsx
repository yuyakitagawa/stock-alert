import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CategoryBadge from "@/components/CategoryBadge";
import DealDirectionBadge from "@/components/DealDirectionBadge";
import ArticleCard from "@/components/ArticleCard";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import { excerptFromHtml, formatDate, formatDealAmountOrCorrection } from "@/lib/format";
import { getArticleDetail, getArticleList } from "@/lib/microcms";
import { SITE_NAME_EN, SITE_URL } from "@/lib/site";
import { UI } from "@/lib/i18n";
import { isIndexableArticle } from "@/lib/articleIndexability";
import AdUnit from "@/components/AdUnit";

// ja版と同じ理由でISRの保持を1日にする（記事本文は公開後ほぼ変わらず、60秒だと
// クローラーのアクセスがほぼ毎回再生成に当たってTTFBが落ちる）。
export const revalidate = 86400;

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
    // 低価値な開示はja版と同じ基準でnoindex（判定はlib/articleIndexability.tsに集約）。
    ...(isIndexableArticle(article) ? {} : { robots: { index: false, follow: true } }),
    alternates: { canonical: url, languages: { ja: `${SITE_URL}/articles/${id}`, en: url } },
    openGraph: {
      type: "article",
      url,
      title: article.titleEn,
      description,
      // 日付はEDINET開示日に一本化（ja版のgenerateMetadataの注記を参照）。
      publishedTime: article.dealDate,
      modifiedTime: article.dealDate,
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
    datePublished: article.dealDate,
    dateModified: article.dealDate,
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
      {/* タイトルが長いとパンくずが複数行になるため1行ellipsisで切る(ja版と同じ)。
          SEO用のBreadcrumbList(JSON-LD)はフルタイトルのまま。 */}
      <nav aria-label={t.breadcrumbAria} className="flex items-center gap-1.5 border-b border-rule px-6 py-3 text-xs text-foreground/50">
        <Link href="/en" className="flex-none hover:text-brand-blue">{t.top}</Link>
        <span aria-hidden>/</span>
        <span className="min-w-0 truncate text-foreground/70">{article.titleEn}</span>
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
          {/* 分類はリンク付きChip(CategoryBadge)へ一本化。ドット版(DealTypeBadge)を併記すると
              同一ラベルが二重表示になっていた(ja版と同じ修正)。 */}
          <CategoryBadge dealType={article.dealType} locale="en" />
          <DealDirectionBadge tags={article.tags} locale="en" />
        </div>
        <h1 className="mb-4 text-2xl font-bold leading-snug text-brand-navy sm:text-3xl">
          {article.titleEn}
        </h1>
        <Box
          component="dl"
          sx={{
            m: 0,
            mb: 4,
            py: 2,
            borderTop: 1,
            borderBottom: 1,
            borderColor: "divider",
            display: "grid",
            gridTemplateColumns: { xs: "repeat(2, 1fr)", sm: "repeat(4, 1fr)" },
            columnGap: 2,
            rowGap: 2,
          }}
        >
          {/* 銘柄名は折り返し不能な長い固有名詞(日本語社名)のため、モバイルの半カラム幅では
              縦割れになる。この項目だけはxsで行全体を使う(ja版と同じ扱い)。 */}
          <Box sx={{ gridColumn: { xs: "1 / -1", sm: "auto" } }}>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>{t.stockLabel}</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
              <Link
                href={`/en/stocks/${article.stockCode}`}
                className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
              >
                {article.stockName} ({article.stockCode}){t.viewHistorySuffix}
              </Link>
            </Typography>
          </Box>
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>{t.dealDateLabel}</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {formatDate(article.dealDate, "en")}
            </Typography>
          </Box>
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>{t.dealSizeLabel}</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {formatDealAmountOrCorrection(article, "en")}
            </Typography>
          </Box>
          {article.sourceUrl && (
            <Box>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>{t.sourceLabel}</Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5 }}>
                <a
                  href={article.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-brand-blue hover:underline"
                >
                  {t.viewOriginal}
                </a>
              </Typography>
            </Box>
          )}
        </Box>
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
            <h2 className="mb-5 text-xl font-bold text-brand-navy">
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
      <AdUnit placement="bottom" locale="en" />
    </article>
  );
}
