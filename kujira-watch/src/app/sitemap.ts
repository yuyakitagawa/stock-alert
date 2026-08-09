import type { MetadataRoute } from "next";
import { getAllArticlesForSitemap, getTranslatedArticlesForSitemap } from "@/lib/microcms";
import { getAllFilers } from "@/lib/investors";
import { SITE_URL } from "@/lib/site";
import { CATEGORIES, DEAL_TYPES } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";

// microCMSの一時的な障害時にビルド自体が失敗しないよう、ビルド時の事前生成を行わず
// リクエスト時に生成する（データ取得自体は60秒のfetchキャッシュが効く）。
export const dynamic = "force-dynamic";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const [articles, filers, translatedArticles] = await Promise.all([
    getAllArticlesForSitemap(),
    getAllFilers(),
    getTranslatedArticlesForSitemap(),
  ]);
  const translatedIds = new Set(translatedArticles.map((a) => a.id));

  const investorEntries: MetadataRoute.Sitemap = filers.map((filer) => ({
    url: `${SITE_URL}/investors/${encodeURIComponent(filer.filerName)}`,
    lastModified: filer.latestDiscDate,
    changeFrequency: "weekly",
    priority: 0.5,
  }));

  const articleEntries: MetadataRoute.Sitemap = articles.map((article) => {
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

  const enArticleEntries: MetadataRoute.Sitemap = translatedArticles.map((article) => ({
    url: `${SITE_URL}/en/articles/${article.id}`,
    lastModified: article.updatedAt,
    changeFrequency: "monthly",
    priority: 0.7,
    alternates: { languages: { ja: `${SITE_URL}/articles/${article.id}`, en: `${SITE_URL}/en/articles/${article.id}` } },
  }));

  const categoryEntries: MetadataRoute.Sitemap = CATEGORIES.map((category) => ({
    url: `${SITE_URL}/category/${encodeURIComponent(category)}`,
    changeFrequency: "daily",
    priority: 0.5,
  }));

  const enCategoryEntries: MetadataRoute.Sitemap = DEAL_TYPES.map((dealType) => ({
    url: `${SITE_URL}/en/category/${DEAL_TYPE_EN[dealType].slug}`,
    changeFrequency: "daily",
    priority: 0.5,
  }));

  const stockCodes = [...new Set(articles.map((article) => article.stockCode).filter(Boolean))];
  const stockEntries: MetadataRoute.Sitemap = stockCodes.map((code) => ({
    url: `${SITE_URL}/stocks/${code}`,
    changeFrequency: "weekly",
    priority: 0.6,
  }));

  const enStockCodes = [
    ...new Set(translatedArticles.map((article) => article.stockCode).filter(Boolean)),
  ];
  const enStockEntries: MetadataRoute.Sitemap = enStockCodes.map((code) => ({
    url: `${SITE_URL}/en/stocks/${code}`,
    changeFrequency: "weekly",
    priority: 0.6,
  }));

  const dealDates = [
    ...new Set(articles.map((article) => article.dealDate.slice(0, 10)).filter(Boolean)),
  ];
  const dateEntries: MetadataRoute.Sitemap = dealDates.map((date) => ({
    url: `${SITE_URL}/date/${date}`,
    changeFrequency: "monthly",
    priority: 0.5,
  }));

  return [
    { url: SITE_URL, changeFrequency: "daily", priority: 1, alternates: { languages: { ja: SITE_URL, en: `${SITE_URL}/en` } } },
    { url: `${SITE_URL}/en`, changeFrequency: "daily", priority: 1, alternates: { languages: { ja: SITE_URL, en: `${SITE_URL}/en` } } },
    { url: `${SITE_URL}/weekly`, changeFrequency: "daily", priority: 0.9 },
    { url: `${SITE_URL}/about`, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` } } },
    { url: `${SITE_URL}/en/about`, changeFrequency: "yearly", priority: 0.3, alternates: { languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` } } },
    { url: `${SITE_URL}/faq`, changeFrequency: "monthly", priority: 0.6 },
    { url: `${SITE_URL}/investors`, changeFrequency: "daily", priority: 0.6 },
    { url: `${SITE_URL}/stocks`, changeFrequency: "daily", priority: 0.6 },
    ...categoryEntries,
    ...enCategoryEntries,
    ...stockEntries,
    ...enStockEntries,
    ...dateEntries,
    ...investorEntries,
    ...articleEntries,
    ...enArticleEntries,
  ];
}
