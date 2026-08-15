import { createClient } from "microcms-js-sdk";
import { unstable_cache } from "next/cache";
import type { AboutPage, Article, ArticleContent, DealType } from "@/types/article";

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

// microCMSの1リクエストあたりの取得上限は100件（APIの仕様）。/weeklyの週間集計は
// 週によっては100件を超える（実測で直近週196件）ため、上限に達したらoffsetページネーションで
// 追加取得し、範囲内の全件を取りこぼさないようにする。
const MAX_PAGE_SIZE = 100;
async function fetchAllArticlesByFilter(filters: string): Promise<ArticleContent[]> {
  const all: ArticleContent[] = [];
  let offset = 0;
  for (;;) {
    const result = await client.getList<Article>({
      endpoint: "articles",
      queries: { filters, orders: "-dealDate,-dealAmount", limit: MAX_PAGE_SIZE, offset },
      customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
    });
    all.push(...result.contents.map(normalizeDealType));
    offset += MAX_PAGE_SIZE;
    if (result.contents.length < MAX_PAGE_SIZE || offset >= (result.totalCount ?? offset)) break;
  }
  return all;
}

// microCMSの`titleEn[exists]true`フィルタは実データがあってもヒットしない既知不具合が
// あるため（2026-08-14確認、begins_with等の他演算子は正常に動く一方でexistsだけ0件を返す。
// 原因はmicroCMS側未特定）、EN側の絞り込みはこのフィルタに頼らず、dealType等の
// 信頼できる条件でサーバー側フィルタした後、titleEn/bodyEnの有無をクライアント側で判定する。
function isTranslated(article: Pick<Article, "titleEn" | "bodyEn">): boolean {
  return Boolean(article.titleEn && article.bodyEn);
}

async function fetchAllArticles(dealType?: DealType): Promise<ArticleContent[]> {
  const contents = await client.getAllContents<Article>({
    endpoint: "articles",
    queries: {
      orders: "-dealDate,-dealAmount",
      filters: dealType ? `dealType[contains]${dealType}` : undefined,
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return contents.map(normalizeDealType);
}

export async function getArticleList(params: {
  offset?: number;
  limit?: number;
  dealType?: DealType;
  translatedOnly?: boolean;
} = {}) {
  const { offset = 0, limit = ARTICLES_PER_PAGE, dealType, translatedOnly = false } = params;

  if (translatedOnly) {
    const all = await fetchAllArticles(dealType);
    const translated = all.filter(isTranslated);
    return { totalCount: translated.length, contents: translated.slice(offset, offset + limit) };
  }

  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      offset,
      limit,
      // 取引日(dealDate)が新しい順、同じ日の中では金額規模(dealAmount)が大きい順。
      orders: "-dealDate,-dealAmount",
      filters: dealType ? `dealType[contains]${dealType}` : undefined,
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  // フィルタが0件にマッチする場合、microCMSがtotalCountを返さないことがある
  // （NaN化してページネーション/オートスクロール側の計算が壊れるのを防ぐ）。
  return { ...result, totalCount: result.totalCount ?? 0, contents: result.contents.map(normalizeDealType) };
}

export const FEATURED_POOL_SIZE = 20;
export const FEATURED_COUNT = 3;

// 「注目」枠: 直近FEATURED_POOL_SIZE件を「日付優先→同日内は金額が大きい順」で取得し、
// 先頭FEATURED_COUNT件を選ぶ。単純な新着1件だと金額の小さい取引が「注目」に出てしまうため
// 同日内では規模の大きい取引を優先しつつ、当日分がある限りは古い日の大型取引に
// 押しのけられないようにする（プール全体を金額だけで並べ替えると、投稿数が少ない日に
// 数日前の大型取引が「注目」を占有し続けてしまうため、この並び替えはしない）。
export async function getFeaturedArticles(
  poolSize = FEATURED_POOL_SIZE,
  count = FEATURED_COUNT,
  translatedOnly = false
) {
  if (translatedOnly) {
    const all = await fetchAllArticles();
    return all.filter(isTranslated).slice(0, poolSize).slice(0, count);
  }
  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      limit: poolSize,
      orders: "-dealDate,-dealAmount",
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return result.contents.map(normalizeDealType).slice(0, count);
}

export async function getArticlesByStockCode(stockCode: string, params: { translatedOnly?: boolean } = {}) {
  const { translatedOnly = false } = params;
  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      filters: `stockCode[equals]${stockCode}`,
      orders: "-dealDate,-dealAmount",
      limit: 100,
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  let contents = result.contents.map(normalizeDealType);
  if (translatedOnly) contents = contents.filter(isTranslated);
  return { ...result, totalCount: contents.length, contents };
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

// 注意: `dealDate[greater_than]YYYY-MM-DD`は日付のみのフィルタ値がJSTとして解釈される
// らしく（dealDateは常にUTC深夜0時で保存）、境界日当日のdealDateも含まれる（実機検証で
// 確認済み: greater_than当日 = equalsした場合と同じ件数がヒットする）。一方`less_than`は
// 境界日当日を含まない。そのため「直近days日間」を暦日ベースでdays日ぴったりにするには、
// カットオフを`days`日前ではなく`days - 1`日前にする必要がある。
export async function getRecentArticles(days: number) {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - (days - 1));
  const cutoffDate = cutoff.toISOString().slice(0, 10);

  const contents = await fetchAllArticlesByFilter(`dealDate[greater_than]${cutoffDate}`);
  return { contents };
}

// /weeklyの「先週比」用。getRecentArticles(days)が取得する直近days暦日の直前・
// 同じ日数分の期間を取得する（境界の扱いはgetRecentArticles上部の注意書きを参照）。
export async function getPreviousPeriodArticles(days: number) {
  const from = new Date();
  from.setDate(from.getDate() - (days * 2 - 1));
  const fromDate = from.toISOString().slice(0, 10);
  const to = new Date();
  to.setDate(to.getDate() - (days - 1));
  const toDate = to.toISOString().slice(0, 10);

  const contents = await fetchAllArticlesByFilter(
    `dealDate[greater_than]${fromDate}[and]dealDate[less_than]${toDate}`
  );
  return { contents };
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
// /stocks は searchParams(sector) を読むため Next.js が自動でdynamic renderingにし、
// ページ単位のrevalidateが効かない。全437件をmicroCMSから100件ずつ取得する
// getAllContents自体もSDK仕様でページ間に1秒スリープが入り単体で4秒以上かかるため、
// unstable_cacheで結果自体をrevalidate間キャッシュしてリクエスト毎の再実行を防ぐ。
export const getAllStocksForIndex = unstable_cache(
  async (): Promise<StockSummary[]> => {
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
  },
  ["all-stocks-for-index"],
  { revalidate: REVALIDATE_SECONDS }
);

export async function getAboutPage() {
  return client.getObject<AboutPage>({
    endpoint: "about",
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
}

// EN側サイトマップ用。titleEn/bodyEn両方がある記事のみ（日英混在ページを出さないため）。
// titleEn[exists]true フィルタが機能しないため、全件取得してクライアント側で絞り込む。
export async function getTranslatedArticlesForSitemap() {
  const contents = await client.getAllContents<
    Pick<Article, "dealType" | "stockCode" | "titleEn" | "bodyEn">
  >({
    endpoint: "articles",
    queries: {
      fields: "id,updatedAt,publishedAt,dealType,stockCode,titleEn,bodyEn",
      orders: "-publishedAt",
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return contents.map(normalizeDealType).filter((a) => a.titleEn && a.bodyEn);
}
