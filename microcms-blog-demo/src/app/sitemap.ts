import type { MetadataRoute } from "next";
import { getAllArticlesForSitemap } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { CATEGORIES } from "@/types/article";

// microCMSの一時的な障害時にビルド自体が失敗しないよう、ビルド時の事前生成を行わず
// リクエスト時に生成する（データ取得自体は60秒のfetchキャッシュが効く）。
export const dynamic = "force-dynamic";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const articles = await getAllArticlesForSitemap();

  const articleEntries: MetadataRoute.Sitemap = articles.map((article) => ({
    url: `${SITE_URL}/articles/${article.id}`,
    lastModified: article.updatedAt,
    changeFrequency: "monthly",
    priority: 0.7,
  }));

  const categoryEntries: MetadataRoute.Sitemap = CATEGORIES.map((category) => ({
    url: `${SITE_URL}/category/${encodeURIComponent(category)}`,
    changeFrequency: "daily",
    priority: 0.5,
  }));

  return [
    { url: SITE_URL, changeFrequency: "daily", priority: 1 },
    { url: `${SITE_URL}/about`, changeFrequency: "yearly", priority: 0.3 },
    ...categoryEntries,
    ...articleEntries,
  ];
}
