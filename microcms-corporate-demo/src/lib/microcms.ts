import { createClient } from "microcms-js-sdk";
import type { NewsContent } from "@/types/news";
import { MOCK_NEWS } from "@/lib/mock-news";

const serviceDomain = process.env.MICROCMS_SERVICE_DOMAIN;
const apiKey = process.env.MICROCMS_API_KEY;

// 実際のmicroCMSサービスを未設定でも `npm run dev` だけでデモを確認できるよう、
// 環境変数が無い場合はモックデータにフォールバックする。
export const USING_MOCK_DATA = !serviceDomain || !apiKey;

const client = USING_MOCK_DATA
  ? null
  : createClient({ serviceDomain: serviceDomain!, apiKey: apiKey! });

export const NEWS_PER_PAGE = 6;
export const REVALIDATE_SECONDS = 60;

export async function getNewsList(params: { offset?: number; limit?: number } = {}) {
  const { offset = 0, limit = NEWS_PER_PAGE } = params;

  if (!client) {
    const contents = MOCK_NEWS.slice(offset, offset + limit);
    return { contents, totalCount: MOCK_NEWS.length };
  }

  return client.getList<NewsContent>({
    endpoint: "news",
    queries: { offset, limit, orders: "-publishedAt" },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
}

export async function getNewsDetail(id: string): Promise<NewsContent> {
  if (!client) {
    const found = MOCK_NEWS.find((news) => news.id === id);
    if (!found) throw new Error(`news not found: ${id}`);
    return found;
  }

  return client.getListDetail<NewsContent>({
    endpoint: "news",
    contentId: id,
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
}
