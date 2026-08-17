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

// 固定ページ・カテゴリ・FAQ・ランキングなど、データ取得なしで列挙できるページ群。
function pageEntries(): MetadataRoute.Sitemap {
  return [
    { url: SITE_URL, changeFrequency: "daily", priority: 1, alternates: { languages: { ja: SITE_URL, en: `${SITE_URL}/en` } } },
    { url: `${SITE_URL}/en`, changeFrequency: "daily", priority: 1, alternates: { languages: { ja: SITE_URL, en: `${SITE_URL}/en` } } },
    { url: `${SITE_URL}/weekly`, changeFrequency: "daily", priority: 0.9 },
    { url: `${SITE_URL}/disclosures`, changeFrequency: "daily", priority: 0.9 },
    { url: `${SITE_URL}/activists`, changeFrequency: "daily", priority: 0.8 },
    { url: `${SITE_URL}/monthly`, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/trending`, changeFrequency: "daily", priority: 0.8 },
    { url: `${SITE_URL}/about`, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` } } },
    { url: `${SITE_URL}/en/about`, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` } } },
    { url: `${SITE_URL}/privacy`, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/privacy`, en: `${SITE_URL}/en/privacy` } } },
    { url: `${SITE_URL}/en/privacy`, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/privacy`, en: `${SITE_URL}/en/privacy` } } },
    { url: `${SITE_URL}/faq`, changeFrequency: "monthly", priority: 0.6 },
    // FAQはカテゴリ別ページにQ&A本文を置いているので、各カテゴリもサイトマップに載せる
    // （ハブの/faqからもリンクしているが、確実に拾わせるため）。
    ...FAQ_CATEGORIES.map((category) => ({
      url: `${SITE_URL}/faq/${category.id}`,
      changeFrequency: "monthly" as const,
      priority: 0.5,
    })),
    { url: `${SITE_URL}/investors`, changeFrequency: "daily", priority: 0.6 },
    { url: `${SITE_URL}/ranking`, changeFrequency: "weekly", priority: 0.7 },
    { url: `${SITE_URL}/ranking/buys`, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/ranking/sells`, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/ranking/filings`, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/ranking/activist`, changeFrequency: "daily", priority: 0.7 },
    { url: `${SITE_URL}/stocks`, changeFrequency: "daily", priority: 0.6 },
    ...CATEGORIES.map((category) => ({
      url: `${SITE_URL}/category/${encodeURIComponent(category)}`,
      changeFrequency: "daily" as const,
      priority: 0.5,
    })),
    ...DEAL_TYPES.map((dealType) => ({
      url: `${SITE_URL}/en/category/${DEAL_TYPE_EN[dealType].slug}`,
      changeFrequency: "daily" as const,
      priority: 0.5,
    })),
  ];
}

async function stockEntries(): Promise<MetadataRoute.Sitemap> {
  const [articles, translatedArticles] = await Promise.all([
    getAllArticlesForSitemap(),
    getTranslatedArticlesForSitemap(),
  ]);
  const stockCodes = [...new Set(articles.map((article) => article.stockCode).filter(Boolean))];
  const enStockCodes = [
    ...new Set(translatedArticles.map((article) => article.stockCode).filter(Boolean)),
  ];
  return [
    ...stockCodes.map((code) => ({
      url: `${SITE_URL}/stocks/${code}`,
      changeFrequency: "weekly" as const,
      priority: 0.6,
    })),
    ...enStockCodes.map((code) => ({
      url: `${SITE_URL}/en/stocks/${code}`,
      changeFrequency: "weekly" as const,
      priority: 0.6,
    })),
  ];
}

async function dateEntries(): Promise<MetadataRoute.Sitemap> {
  const articles = await getAllArticlesForSitemap();
  const dealDates = [
    ...new Set(articles.map((article) => article.dealDate.slice(0, 10)).filter(Boolean)),
  ];
  // 月別アーカイブ。取引日別ページの親ハブなので、日別ページより優先度を高くする。
  const months = [...new Set(dealDates.map((date) => date.slice(0, 7)))];
  return [
    ...months.map((month) => ({
      url: `${SITE_URL}/monthly/${month}`,
      changeFrequency: "weekly" as const,
      priority: 0.6,
    })),
    ...dealDates.map((date) => ({
      url: `${SITE_URL}/date/${date}`,
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
