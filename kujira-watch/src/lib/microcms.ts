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

export const FEATURED_POOL_SIZE = 20;
export const FEATURED_COUNT = 3;

// 「注目」枠: 直近FEATURED_POOL_SIZE件の中から取得金額(dealAmount)が大きい順にFEATURED_COUNT件を選ぶ。
// 単純な新着1件だと金額の小さい取引が「注目」に出てしまうため、直近プールの中で規模の大きい
// 取引を優先する。
export async function getFeaturedArticles(poolSize = FEATURED_POOL_SIZE, count = FEATURED_COUNT) {
  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      limit: poolSize,
      orders: "-dealDate,-dealAmount",
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  const contents = result.contents.map(normalizeDealType);
  return [...contents].sort((a, b) => b.dealAmount - a.dealAmount).slice(0, count);
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

// dealDateはweb/publish_blog_articles.pyが`${disc_date}T00:00:00.000Z`形式(UTC深夜0時)で
// 保存しているため、日付部分(YYYY-MM-DD)から同じ形式を組み立ててequalsで完全一致させる。
export async function getArticlesByDealDate(date: string) {
  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      filters: `dealDate[equals]${date}T00:00:00.000Z`,
      orders: "-dealAmount",
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

export async function getRecentArticles(days: number) {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - days);
  const cutoffDate = cutoff.toISOString().slice(0, 10);

  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      filters: `dealDate[greater_than]${cutoffDate}`,
      orders: "-dealDate,-dealAmount",
      limit: 100,
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return { ...result, contents: result.contents.map(normalizeDealType) };
}

export type StockSearchResult = { stockCode: string; stockName: string };

// ヘッダーの検索ボックス用。企業名・証券コードの部分一致で記事を引き、
// 銘柄(stockCode)単位で重複排除して返す。
export async function searchStocks(keyword: string): Promise<StockSearchResult[]> {
  const q = keyword.trim();
  if (!q) return [];

  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      filters: `stockCode[contains]${q}[or]stockName[contains]${q}`,
      fields: "stockCode,stockName",
      orders: "-dealDate",
      limit: 100,
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });

  const seen = new Map<string, string>();
  for (const { stockCode, stockName } of result.contents) {
    if (!seen.has(stockCode)) seen.set(stockCode, stockName);
  }
  return Array.from(seen, ([stockCode, stockName]) => ({ stockCode, stockName })).slice(0, 20);
}

export async function getAllArticlesForSitemap() {
  const contents = await client.getAllContents<
    Pick<Article, "dealType" | "stockCode" | "dealDate">
  >({
    endpoint: "articles",
    queries: {
      fields: "id,updatedAt,publishedAt,dealType,stockCode,dealDate",
      orders: "-publishedAt",
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return contents.map(normalizeDealType);
}

export type StockSummary = { stockCode: string; stockName: string; articleCount: number; latestDealDate: string };

// /stocks（銘柄一覧）用。記事が1件以上ある銘柄をstockCode単位で集約する。
export async function getAllStocksForIndex(): Promise<StockSummary[]> {
  const contents = await client.getAllContents<Pick<Article, "stockCode" | "stockName" | "dealDate">>({
    endpoint: "articles",
    queries: {
      fields: "stockCode,stockName,dealDate",
      orders: "-dealDate",
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });

  const byCode = new Map<string, StockSummary>();
  for (const { stockCode, stockName, dealDate } of contents) {
    const existing = byCode.get(stockCode);
    if (!existing) {
      byCode.set(stockCode, { stockCode, stockName, articleCount: 1, latestDealDate: dealDate });
    } else {
      existing.articleCount += 1;
    }
  }
  // 一覧は「見て探す」用途のため、更新順ではなく証券コード昇順（辞書的に引ける順番）にする。
  return Array.from(byCode.values()).sort((a, b) => a.stockCode.localeCompare(b.stockCode));
}
