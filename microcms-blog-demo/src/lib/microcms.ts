import { createClient } from "microcms-js-sdk";
import type { Article } from "@/types/article";

const serviceDomain = process.env.MICROCMS_SERVICE_DOMAIN;
const apiKey = process.env.MICROCMS_API_KEY;

if (!serviceDomain || !apiKey) {
  throw new Error(
    "MICROCMS_SERVICE_DOMAIN と MICROCMS_API_KEY を .env.local に設定してください"
  );
}

export const client = createClient({ serviceDomain, apiKey });

export const ARTICLES_PER_PAGE = 10;

export const REVALIDATE_SECONDS = 60;

// dealType/category は microCMS 側で複数選択(配列)設定になっている記事が混在するため、
// 配列で返ってきた場合は先頭要素に正規化する（単一選択の記事はそのまま通す）。
function normalizeSelectFields<T extends { dealType: unknown; category: unknown }>(
  article: T
): T {
  return {
    ...article,
    dealType: Array.isArray(article.dealType) ? article.dealType[0] : article.dealType,
    category: Array.isArray(article.category) ? article.category[0] : article.category,
  };
}

export async function getArticleList(params: {
  offset?: number;
  limit?: number;
  category?: string;
} = {}) {
  const { offset = 0, limit = ARTICLES_PER_PAGE, category } = params;

  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      offset,
      limit,
      orders: "-publishedAt",
      // category は記事によって単一文字列/配列のいずれで格納されているか揺れがあるため、
      // 単一値一致(equals)と配列内包含(contains)の両方をORで見て取りこぼしを防ぐ
      ...(category
        ? { filters: `category[equals]${category}[or]category[contains]${category}` }
        : {}),
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return { ...result, contents: result.contents.map(normalizeSelectFields) };
}

export async function getArticleDetail(id: string) {
  const article = await client.getListDetail<Article>({
    endpoint: "articles",
    contentId: id,
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return normalizeSelectFields(article);
}
