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

export function getArticleList(params: {
  offset?: number;
  limit?: number;
  category?: string;
} = {}) {
  const { offset = 0, limit = ARTICLES_PER_PAGE, category } = params;

  return client.getList<Article>({
    endpoint: "articles",
    queries: {
      offset,
      limit,
      orders: "-publishedAt",
      ...(category ? { filters: `category[equals]${category}` } : {}),
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
}

export function getArticleDetail(id: string) {
  return client.getListDetail<Article>({
    endpoint: "articles",
    contentId: id,
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
}
