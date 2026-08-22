import { createClient } from "microcms-js-sdk";
import { unstable_cache } from "next/cache";
import type { AboutPage, Article, ArticleContent, DealType } from "@/types/article";
import { displayText } from "./format";

const serviceDomain = process.env.MICROCMS_SERVICE_DOMAIN;
const apiKey = process.env.MICROCMS_API_KEY;

if (!serviceDomain || !apiKey) {
  throw new Error(
    "MICROCMS_SERVICE_DOMAIN と MICROCMS_API_KEY を .env.local に設定してください"
  );
}

// retry: 429(レート制限)・5xxを指数バックオフで最大2回再試行する（SDK標準機能）。
// 事前生成が1,000ページ超になりビルド時の並列fetchがmicroCMSの秒間上限に当たるため必須
// （2026-08-22、retry無しでは本番ビルドが429で失敗した実測あり）。
export const client = createClient({ serviceDomain, apiKey, retry: true });

export const ARTICLES_PER_PAGE = 10;

export const REVALIDATE_SECONDS = 60;

// dealType は microCMS 側で複数選択(配列)設定になっている記事が混在するため、
// 配列で返ってきた場合は先頭要素に正規化する（単一選択の記事はそのまま通す）。
// あわせてタイトルの全角英数字（EDINETの提出者名がＯａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ…の形で
// 登録されており、そこから組み立てたタイトルも全角のまま）を半角へ寄せる。一覧・見出し・
// metadata・共有テキストのどこで見ても同じ表記になるよう、受け取った時点で1か所で行う
// （本文HTMLはlinkifyFilerNamesがDB上の正式表記と突合するため変換しない）。
function normalizeDealType<T extends { dealType: unknown }>(article: T): T {
  const withTitle = article as T & { title?: unknown };
  return {
    ...article,
    dealType: Array.isArray(article.dealType) ? article.dealType[0] : article.dealType,
    ...(typeof withTitle.title === "string" ? { title: displayText(withTitle.title) } : {}),
  };
}

// microCMSの1リクエストあたりの取得上限は100件（APIの仕様）。/weeklyの週間集計は
// 週によっては100件を超える（実測で直近週196件）ため、上限に達したらoffsetページネーションで
// 追加取得し、範囲内の全件を取りこぼさないようにする。
const MAX_PAGE_SIZE = 100;
// microCMS公式SDKのgetAllContentsはページ間に1秒の固定スリープが入る仕様で、
// 数百件の全件取得に数秒かかる。全件走査が必要な集計（銘柄一覧・月別アーカイブ）では
// 1ページ目のtotalCountから残りのoffsetを割り出し、並列に取得して待ち時間を潰す。
async function fetchAllPagesParallel<T>(queries: { fields: string; orders: string; filters?: string }): Promise<T[]> {
  const requestInit = { next: { revalidate: REVALIDATE_SECONDS } };
  const first = await client.getList<T>({
    endpoint: "articles",
    queries: { ...queries, limit: MAX_PAGE_SIZE, offset: 0 },
    customRequestInit: requestInit,
  });

  const remainingOffsets: number[] = [];
  for (let offset = MAX_PAGE_SIZE; offset < first.totalCount; offset += MAX_PAGE_SIZE) {
    remainingOffsets.push(offset);
  }
  const restPages = await Promise.all(
    remainingOffsets.map((offset) =>
      client.getList<T>({
        endpoint: "articles",
        queries: { ...queries, limit: MAX_PAGE_SIZE, offset },
        customRequestInit: requestInit,
      })
    )
  );
  return [first, ...restPages].flatMap((page) => page.contents);
}

async function fetchAllArticlesByFilter(filters: string): Promise<ArticleContent[]> {
  const queries = { filters, orders: "-dealDate,-dealAmount" };
  const requestInit = { next: { revalidate: REVALIDATE_SECONDS } };

  const first = await client.getList<Article>({
    endpoint: "articles",
    queries: { ...queries, limit: MAX_PAGE_SIZE, offset: 0 },
    customRequestInit: requestInit,
  });

  // 1ページ目のtotalCountから残りのoffsetを割り出して並列取得する。
  // 直列に繰り返すと1リクエストぶんの往復（実測で200〜300ms）がページ数だけ積み上がり、
  // 365件あった2026年8月の/monthlyがTTFB約1秒になっていた。
  const remainingOffsets: number[] = [];
  for (let offset = MAX_PAGE_SIZE; offset < (first.totalCount ?? 0); offset += MAX_PAGE_SIZE) {
    remainingOffsets.push(offset);
  }
  const restPages = await Promise.all(
    remainingOffsets.map((offset) =>
      client.getList<Article>({
        endpoint: "articles",
        queries: { ...queries, limit: MAX_PAGE_SIZE, offset },
        customRequestInit: requestInit,
      })
    )
  );
  return [first, ...restPages].flatMap((page) => page.contents.map(normalizeDealType));
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

// 「注目」枠: 直近FEATURED_POOL_SIZE件のプール（日付優先→同日内は金額が大きい順で取得）
// から、推定取引金額が大きい順に先頭FEATURED_COUNT件を選ぶ。
// プールの取得を日付優先にしているのは、金額だけで並べ替えると投稿数が少ない日に
// 数日前の大型取引が「注目」を占有し続けてしまうため。
// web/publish_blog_articles.pyのget_featured_article_ids()と同じロジック。
function pickFeatured(pool: ArticleContent[], count: number): ArticleContent[] {
  return [...pool].sort((a, b) => b.dealAmount - a.dealAmount).slice(0, count);
}

export async function getFeaturedArticles(
  poolSize = FEATURED_POOL_SIZE,
  count = FEATURED_COUNT,
  translatedOnly = false
) {
  if (translatedOnly) {
    const all = await fetchAllArticles();
    return pickFeatured(all.filter(isTranslated).slice(0, poolSize), count);
  }
  const result = await client.getList<Article>({
    endpoint: "articles",
    queries: {
      limit: poolSize,
      orders: "-dealDate,-dealAmount",
    },
    customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
  });
  return pickFeatured(result.contents.map(normalizeDealType), count);
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

export type ArticleDigest = Pick<Article, "dealDate" | "dealAmount" | "tags" | "dealType">;

// /weeklyの週次トレンド用。直近days暦日の記事を集計に必要な4フィールドだけで取得する
// （全フィールドだと本文込みで数百件が重くなるため）。dealTypeは投資家分類別の週次トレンド用。
export async function getRecentArticleDigests(days: number): Promise<ArticleDigest[]> {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - (days - 1));
  const cutoffDate = cutoff.toISOString().slice(0, 10);
  const digests = await fetchAllPagesParallel<ArticleDigest>({
    fields: "dealDate,dealAmount,tags,dealType",
    orders: "-dealDate",
    filters: `dealDate[greater_than]${cutoffDate}`,
  });
  // dealTypeが複数選択(配列)で入っている記事が混在するため、他の取得経路と同じ形に正規化する。
  return digests.map(normalizeDealType);
}

export const MONTH_PATTERN = /^\d{4}-\d{2}$/;

// /monthly/[month]用。YYYY-MM の暦月に含まれる記事を全件取得する。
// 境界の扱いはgetRecentArticles上部の注意書きの通りで、`greater_than`は境界日当日を
// 含み`less_than`は含まないため、月初日〜翌月初日を指定するとちょうど1か月分になる。
export async function getArticlesByMonth(month: string) {
  const [year, mon] = month.split("-").map(Number);
  const firstDay = `${month}-01`;
  const next = new Date(Date.UTC(year, mon, 1)); // mon は1始まりなのでこれで翌月1日
  const nextFirstDay = next.toISOString().slice(0, 10);

  const contents = await fetchAllArticlesByFilter(
    `dealDate[greater_than]${firstDay}[and]dealDate[less_than]${nextFirstDay}`
  );
  return { contents };
}

export type MonthSummary = { month: string; count: number; amount: number };

// /monthly（月別アーカイブ一覧）とサイトマップ用。記事が1件以上ある月を新しい順に返す。
// getAllStocksForIndexと同じくSDKのgetAllContents（ページ間に1秒の固定スリープが入る仕様）
// を避け、ページを並列取得する。
export const getAllMonthsForIndex = unstable_cache(
  async (): Promise<MonthSummary[]> => {
    const contents = await fetchAllPagesParallel<Pick<Article, "dealDate" | "dealAmount">>({
      fields: "dealDate,dealAmount",
      orders: "-dealDate",
    });

    const byMonth = new Map<string, MonthSummary>();
    for (const { dealDate, dealAmount } of contents) {
      const month = dealDate.slice(0, 7);
      const existing = byMonth.get(month);
      if (existing) {
        existing.count += 1;
        existing.amount += dealAmount;
      } else {
        byMonth.set(month, { month, count: 1, amount: dealAmount });
      }
    }
    return [...byMonth.values()].sort((a, b) => b.month.localeCompare(a.month));
  },
  ["all-months-for-index"],
  { revalidate: REVALIDATE_SECONDS }
);

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

// サイトマップ用の軽量な記事参照。sitemapはforce-dynamicで毎リクエスト実行されるため、
// microCMS全件ページングを毎回行わないようunstable_cacheで1時間持つ。
// dealTypeはカテゴリページの<lastmod>（カテゴリ内の記事の最新updatedAt）算出に使う。
export type SitemapArticleRef = {
  id: string;
  stockCode: string;
  dealDate: string;
  dealType: DealType;
  dealAmount: number;
  ratioChangePct?: number;
  // 同一「銘柄×提出者」の記事を1本に絞る判定(supersededArticleIds)に使う。
  filerName?: string;
};

export const getAllArticlesForSitemap = unstable_cache(
  async (): Promise<SitemapArticleRef[]> => {
    const contents = await client.getAllContents<
      Pick<Article, "stockCode" | "dealDate" | "dealType" | "dealAmount" | "ratioChangePct" | "filerName">
    >({
      endpoint: "articles",
      queries: {
        fields: "id,stockCode,dealDate,dealType,dealAmount,ratioChangePct,filerName",
        orders: "-publishedAt",
      },
      customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
    });
    return contents
      .map(normalizeDealType)
      .map(({ id, stockCode, dealDate, dealType, dealAmount, ratioChangePct, filerName }) => ({
        id,
        stockCode,
        dealDate,
        dealType,
        dealAmount,
        ratioChangePct,
        filerName,
      }));
  },
  ["sitemap-articles"],
  { revalidate: 3600 }
);

export type StockSummary = { stockCode: string; stockName: string; articleCount: number; latestDealDate: string };

// /stocks（銘柄一覧）用。記事が1件以上ある銘柄をstockCode単位で集約する。
// /stocks は searchParams(sector) を読むため Next.js が自動でdynamic renderingにし、
// ページ単位のrevalidateが効かない。またVercel上のData Cacheはsearchparam違いの
// リクエスト間で共有される保証がない（実測でsector絞り込みURLはキャッシュヒットしなかった）ため、
// unstable_cacheだけに頼らず取得処理自体を速くする（fetchAllPagesParallelの並列取得）。
type StockIndexRow = Pick<Article, "stockCode" | "stockName" | "dealDate">;

export const getAllStocksForIndex = unstable_cache(
  async (): Promise<StockSummary[]> => {
    const contents = await fetchAllPagesParallel<StockIndexRow>({
      fields: "stockCode,stockName,dealDate",
      orders: "-dealDate",
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
// bodyEn全文を含む重い取得（実測でsitemap応答9秒超の主因）なので、存在判定に使った後は
// 捨てて軽量な参照だけをunstable_cacheに1時間載せる。
export const getTranslatedArticlesForSitemap = unstable_cache(
  async (): Promise<SitemapArticleRef[]> => {
    const contents = await client.getAllContents<
      Pick<
        Article,
        | "stockCode"
        | "dealDate"
        | "dealType"
        | "dealAmount"
        | "ratioChangePct"
        | "filerName"
        | "titleEn"
        | "bodyEn"
      >
    >({
      endpoint: "articles",
      queries: {
        fields: "id,stockCode,dealDate,dealType,dealAmount,ratioChangePct,filerName,titleEn,bodyEn",
        orders: "-publishedAt",
      },
      customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
    });
    return contents
      .filter((a) => a.titleEn && a.bodyEn)
      .map(normalizeDealType)
      .map(({ id, stockCode, dealDate, dealType, dealAmount, ratioChangePct, filerName }) => ({
        id,
        stockCode,
        dealDate,
        dealType,
        dealAmount,
        ratioChangePct,
        filerName,
      }));
  },
  ["sitemap-translated-articles"],
  { revalidate: 3600 }
);

// /aboutの<lastmod>用。microCMSのaboutオブジェクト自体の更新日時だけを取る
// （本文フィールドは不要なのでupdatedAtのみ要求）。
export const getAboutUpdatedAtForSitemap = unstable_cache(
  async (): Promise<string | undefined> => {
    const about = await client.getObject<{ updatedAt?: string }>({
      endpoint: "about",
      queries: { fields: "updatedAt" },
      customRequestInit: { next: { revalidate: REVALIDATE_SECONDS } },
    });
    return about.updatedAt;
  },
  ["sitemap-about-updated-at"],
  { revalidate: 3600 }
);
