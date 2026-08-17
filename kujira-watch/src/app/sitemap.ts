import type { MetadataRoute } from "next";
import { getAllArticlesForSitemap, getTranslatedArticlesForSitemap } from "@/lib/microcms";
import { getAllFilers } from "@/lib/investors";
import { SITE_URL, SITEMAP_IDS, type SitemapId } from "@/lib/site";
import { CATEGORIES, DEAL_TYPES } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import { FAQ_CATEGORIES } from "@/lib/faqData";

// microCMSの一時的な障害時にビルド自体が失敗しないよう、ビルド時の事前生成を行わず
// リクエスト時に生成する（データ取得はunstable_cacheで1時間キャッシュされるため軽い）。
export const dynamic = "force-dynamic";

// URL総数が6,000件を超えたため、/sitemap.xml（sitemapindex, app/sitemap.xml/route.ts）から
// 参照される子サイトマップ /sitemap/<id>.xml に種類別で分割している。
export function generateSitemaps(): { id: SitemapId }[] {
  return SITEMAP_IDS.map((id) => ({ id }));
}

// 記事更新日時(ISO文字列)の最大値。全エントリに<lastmod>を出すための共通ヘルパー
// （同一形式のISO文字列同士なので文字列比較で新しい方が選べる）。
function maxUpdatedAt(updatedAts: string[]): string | undefined {
  let latest: string | undefined;
  for (const u of updatedAts) {
    if (u && (!latest || u > latest)) latest = u;
  }
  return latest;
}

// 固定ページ・カテゴリ・FAQ・ランキングなど。ページ自体の更新日時は追跡していないため、
// <lastmod>にはサイト全体の最新記事更新日時を入れる（ハブページは新着記事で内容が変わるため）。
async function pageEntries(): Promise<MetadataRoute.Sitemap> {
  const articles = await getAllArticlesForSitemap();
  const lastModified = maxUpdatedAt(articles.map((a) => a.updatedAt));
  return [
    { url: SITE_URL, lastModified, changeFrequency: "daily", priority: 1, alternates: { languages: { ja: SITE_URL, en: `${SITE_URL}/en` } } },
    { url: `${SITE_URL}/en`, lastModified, changeFrequency: "daily", priority: 1, alternates: { languages: { ja: SITE_URL, en: `${SITE_URL}/en` } } },
    { url: `${SITE_URL}/weekly`, lastModified, changeFrequency: "daily", priority: 0.9 },
    { url: `${SITE_URL}/disclosures`, lastModified, changeFrequency: "daily", priority: 0.9 },
    { url: `${SITE_URL}/activists`, lastModified, changeFrequency: "daily", priority: 0.8 },
    { url: `${SITE_URL}/monthly`, lastModified, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/trending`, lastModified, changeFrequency: "daily", priority: 0.8 },
    { url: `${SITE_URL}/about`, lastModified, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` } } },
    { url: `${SITE_URL}/en/about`, lastModified, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` } } },
    { url: `${SITE_URL}/privacy`, lastModified, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/privacy`, en: `${SITE_URL}/en/privacy` } } },
    { url: `${SITE_URL}/en/privacy`, lastModified, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/privacy`, en: `${SITE_URL}/en/privacy` } } },
    { url: `${SITE_URL}/faq`, lastModified, changeFrequency: "monthly", priority: 0.6 },
    // FAQはカテゴリ別ページにQ&A本文を置いているので、各カテゴリもサイトマップに載せる
    // （ハブの/faqからもリンクしているが、確実に拾わせるため）。
    ...FAQ_CATEGORIES.map((category) => ({
      url: `${SITE_URL}/faq/${category.id}`,
      lastModified,
      changeFrequency: "monthly" as const,
      priority: 0.5,
    })),
    { url: `${SITE_URL}/investors`, lastModified, changeFrequency: "daily", priority: 0.6 },
    { url: `${SITE_URL}/ranking`, lastModified, changeFrequency: "weekly", priority: 0.7 },
    { url: `${SITE_URL}/ranking/buys`, lastModified, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/ranking/sells`, lastModified, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/ranking/filings`, lastModified, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/ranking/activist`, lastModified, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/stocks`, lastModified, changeFrequency: "daily", priority: 0.6 },
    ...CATEGORIES.map((category) => ({
      url: `${SITE_URL}/category/${encodeURIComponent(category)}`,
      lastModified,
      changeFrequency: "daily" as const,
      priority: 0.5,
    })),
    ...DEAL_TYPES.map((dealType) => ({
      url: `${SITE_URL}/en/category/${DEAL_TYPE_EN[dealType].slug}`,
      lastModified,
      changeFrequency: "daily" as const,
      priority: 0.5,
    })),
  ];
}

// キー（銘柄コード・取引日など）ごとの記事の最新updatedAt。
// 銘柄・日別・月別ページの<lastmod>に使う。
function latestUpdatedAtBy<T extends { updatedAt: string }>(
  items: T[],
  keyOf: (item: T) => string
): Map<string, string> {
  const latest = new Map<string, string>();
  for (const item of items) {
    const key = keyOf(item);
    if (!key) continue;
    const prev = latest.get(key);
    if (!prev || item.updatedAt > prev) latest.set(key, item.updatedAt);
  }
  return latest;
}

async function stockEntries(): Promise<MetadataRoute.Sitemap> {
  const [articles, translatedArticles] = await Promise.all([
    getAllArticlesForSitemap(),
    getTranslatedArticlesForSitemap(),
  ]);
  const latestByStock = latestUpdatedAtBy(articles, (a) => a.stockCode);
  const latestByEnStock = latestUpdatedAtBy(translatedArticles, (a) => a.stockCode);
  return [
    ...[...latestByStock.entries()].map(([code, lastModified]) => ({
      url: `${SITE_URL}/stocks/${code}`,
      lastModified,
      changeFrequency: "weekly" as const,
      priority: 0.6,
    })),
    ...[...latestByEnStock.entries()].map(([code, lastModified]) => ({
      url: `${SITE_URL}/en/stocks/${code}`,
      lastModified,
      changeFrequency: "weekly" as const,
      priority: 0.6,
    })),
  ];
}

async function dateEntries(): Promise<MetadataRoute.Sitemap> {
  const articles = await getAllArticlesForSitemap();
  const latestByDate = latestUpdatedAtBy(articles, (a) => a.dealDate.slice(0, 10));
  const latestByMonth = latestUpdatedAtBy(articles, (a) => a.dealDate.slice(0, 7));
  // 月別アーカイブ。取引日別ページの親ハブなので、日別ページより優先度を高くする。
  return [
    ...[...latestByMonth.entries()].map(([month, lastModified]) => ({
      url: `${SITE_URL}/monthly/${month}`,
      lastModified,
      changeFrequency: "weekly" as const,
      priority: 0.6,
    })),
    ...[...latestByDate.entries()].map(([date, lastModified]) => ({
      url: `${SITE_URL}/date/${date}`,
      lastModified,
      changeFrequency: "monthly" as const,
      priority: 0.5,
    })),
  ];
}

async function investorEntries(): Promise<MetadataRoute.Sitemap> {
  const filers = await getAllFilers();
  return filers.map((filer) => ({
    url: `${SITE_URL}/investors/${encodeURIComponent(filer.filerName)}`,
    lastModified: filer.latestDiscDate,
    changeFrequency: "weekly",
    priority: 0.5,
  }));
}

async function articleEntries(): Promise<MetadataRoute.Sitemap> {
  const [articles, translatedArticles] = await Promise.all([
    getAllArticlesForSitemap(),
    getTranslatedArticlesForSitemap(),
  ]);
  const translatedIds = new Set(translatedArticles.map((a) => a.id));
  return articles.map((article) => {
    const hasEn = translatedIds.has(article.id);
    return {
      url: `${SITE_URL}/articles/${article.id}`,
      lastModified: article.updatedAt,
      changeFrequency: "monthly",
      priority: 0.7,
      ...(hasEn
        ? { alternates: { languages: { ja: `${SITE_URL}/articles/${article.id}`, en: `${SITE_URL}/en/articles/${article.id}` } } }
        : {}),
    };
  });
}

async function enArticleEntries(): Promise<MetadataRoute.Sitemap> {
  const translatedArticles = await getTranslatedArticlesForSitemap();
  return translatedArticles.map((article) => ({
    url: `${SITE_URL}/en/articles/${article.id}`,
    lastModified: article.updatedAt,
    changeFrequency: "monthly",
    priority: 0.7,
    alternates: { languages: { ja: `${SITE_URL}/articles/${article.id}`, en: `${SITE_URL}/en/articles/${article.id}` } },
  }));
}

export default async function sitemap(props: {
  id: Promise<string>;
}): Promise<MetadataRoute.Sitemap> {
  const id = (await props.id) as SitemapId;
  switch (id) {
    case "pages":
      return pageEntries();
    case "stocks":
      return stockEntries();
    case "dates":
      return dateEntries();
    case "investors":
      return investorEntries();
    case "articles":
      return articleEntries();
    case "articles-en":
      return enArticleEntries();
  }
}
