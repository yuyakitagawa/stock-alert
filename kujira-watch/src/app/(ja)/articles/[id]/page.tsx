import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CategoryBadge from "@/components/CategoryBadge";
import DealDirectionBadge from "@/components/DealDirectionBadge";
import DealTypeBadge from "@/components/DealTypeBadge";
import ArticleCard from "@/components/ArticleCard";
import ShareButtons from "@/components/ShareButtons";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { excerptFromHtml, formatDate, formatDealAmount, linkifyFilerNames } from "@/lib/format";
import {
  getArticleDetail,
  getArticleList,
  getArticlesByFilerName,
  getArticlesByStockCode,
} from "@/lib/microcms";
import { getFilersByStockCode } from "@/lib/investors";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { categoryLabel } from "@/types/article";
import type { ArticleContent } from "@/types/article";

// Route segment config requires a literal value (cannot import from lib/microcms).
export const revalidate = 60;

// 「同じ銘柄」「同じ投資家」の関連リンクの表示件数。カード表示の関連記事（同じ分類）と
// 違って一行リンクで並べるため、多すぎない範囲で回遊先を増やす。
const RELATED_COUNT = 3;

// 関連リンクの一行表示（取引日・金額つき）。回遊導線なのでカードより密度を優先する。
function RelatedArticleLinks({ articles }: { articles: ArticleContent[] }) {
  return (
    <ul className="divide-y divide-rule border-y border-rule">
      {articles.map((related) => (
        <li key={related.id}>
          <Link
            href={`/articles/${related.id}`}
            className="group flex items-baseline justify-between gap-4 py-3"
          >
            <span className="text-sm font-medium text-brand-navy group-hover:text-brand-blue group-hover:underline">
              {related.title}
            </span>
            <span className="shrink-0 text-xs text-foreground/50">
              {formatDate(related.dealDate)}・{formatDealAmount(related.dealAmount)}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}

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

  // 関連リンクは「同じ銘柄」→「同じ投資家」→「同じ分類」の順に近いものから並べる。
  // 同じ記事・既に上のセクションで出した記事は重複させない。
  const [{ contents: sameCategoryArticles }, { contents: sameStockArticles }, sameFilerArticles, filers] =
    await Promise.all([
      article.dealType ? getArticleList({ dealType: article.dealType, limit: 5 }) : Promise.resolve({ contents: [] }),
      getArticlesByStockCode(article.stockCode),
      article.filerName ? getArticlesByFilerName(article.filerName, RELATED_COUNT + 1) : Promise.resolve([]),
      getFilersByStockCode(article.stockCode),
    ]);

  const relatedStockArticles = sameStockArticles.filter((a) => a.id !== id).slice(0, RELATED_COUNT);
  const shownIds = new Set([id, ...relatedStockArticles.map((a) => a.id)]);
  const relatedFilerArticles = sameFilerArticles.filter((a) => !shownIds.has(a.id)).slice(0, RELATED_COUNT);
  for (const a of relatedFilerArticles) shownIds.add(a.id);
  const relatedArticles = sameCategoryArticles.filter((a) => !shownIds.has(a.id)).slice(0, 4);

  const linkedBody = linkifyFilerNames(article.body, filers.map((f) => f.filerName));

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
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>銘柄</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
              <Link
                href={`/stocks/${article.stockCode}`}
                className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
              >
                {article.stockName}（{article.stockCode}）
              </Link>
            </Typography>
          </Box>
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>取引日</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {formatDate(article.dealDate)}
            </Typography>
          </Box>
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>金額規模</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {formatDealAmount(article.dealAmount)}
            </Typography>
          </Box>
          {article.filerName && (
            <Box>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>取引企業</Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
                <Link
                  href={`/investors/${encodeURIComponent(article.filerName)}`}
                  className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
                >
                  {article.filerName}
                </Link>
              </Typography>
            </Box>
          )}
          {article.sourceUrl && (
            <Box>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>出典</Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5 }}>
                <a
                  href={article.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-brand-blue hover:underline"
                >
                  元記事を見る
                </a>
              </Typography>
            </Box>
          )}
        </Box>
        <div
          className="prose max-w-none prose-headings:text-brand-navy prose-a:text-brand-blue first:prose-p:first-letter:float-left first:prose-p:first-letter:mr-2 first:prose-p:first-letter:text-5xl first:prose-p:first-letter:font-bold first:prose-p:first-letter:text-brand-navy"
          dangerouslySetInnerHTML={{ __html: linkedBody }}
        />
        {tags && tags.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-x-3 gap-y-1 border-t border-rule pt-4 text-xs text-foreground/50">
            {tags.map((tag) => (
              <span key={tag}>#{tag}</span>
            ))}
          </div>
        )}
        <ShareButtons url={url} title={article.title} />
        {relatedStockArticles.length > 0 && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-4 text-lg font-bold text-brand-navy">
              {article.stockName}（{article.stockCode}）の他の記事
            </h2>
            <RelatedArticleLinks articles={relatedStockArticles} />
            <Link
              href={`/stocks/${article.stockCode}`}
              className="kicker mt-3 inline-block text-brand-blue hover:underline"
            >
              この銘柄の大量保有・自社株買い履歴をすべて見る ›
            </Link>
          </div>
        )}
        {article.filerName && relatedFilerArticles.length > 0 && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-4 text-lg font-bold text-brand-navy">
              {article.filerName}の他の記事
            </h2>
            <RelatedArticleLinks articles={relatedFilerArticles} />
            <Link
              href={`/investors/${encodeURIComponent(article.filerName)}`}
              className="kicker mt-3 inline-block text-brand-blue hover:underline"
            >
              この投資家の保有銘柄・取引履歴を見る ›
            </Link>
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
