import { createClient } from "microcms-js-sdk";
import type { Article, DealType } from "@/types/article";

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

// dealType は microCMS 側で複数選択(配列)設定になっている記事が混在するため、
// 配列で返ってきた場合は先頭要素に正規化する（単一選択の記事はそのまま通す）。
function normalizeDealType<T extends { dealType: unknown }>(article: T): T {
  return {
    ...article,
    dealType: Array.isArray(article.dealType) ? article.dealType[0] : article.dealType,
  };
}

export async function getArticleList(params: {
  offset?: number;
  limit?: number;
  dealType?: DealType;
} = {}) {
  const { offset = 0, limit = ARTICLES_PER_PAGE, dealType } = params;

  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      offset,
      limit,
      // 取引日(dealDate)が新しい順、同じ日の中では金額規模(dealAmount)が大きい順。
      orders: "-dealDate,-dealAmount",
      ...(dealType ? { filters: `dealType[contains]${dealType}` } : {}),
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return { ...result, contents: result.contents.map(normalizeDealType) };
}

export async function getArticlesByStockCode(stockCode: string) {
  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      filters: `stockCode[equals]${stockCode}`,
      orders: "-dealDate,-dealAmount",
      limit: 100,
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return { ...result, contents: result.contents.map(normalizeDealType) };
}

export async function getArticleDetail(id: string) {
  const article = await client.getListDetail<Article>({
    endpoint: "articles",
    contentId: id,
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return normalizeDealType(article);
}

export async function getAllArticlesForSitemap() {
  const contents = await client.getAllContents<Pick<Article, "dealType" | "stockCode">>({
    endpoint: "articles",
    queries: { fields: "id,updatedAt,publishedAt,dealType,stockCode", orders: "-publishedAt" },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return contents.map(normalizeDealType);
}
