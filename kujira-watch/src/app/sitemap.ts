import type { MetadataRoute } from "next";
import {
  getAboutUpdatedAtForSitemap,
  getAllArticlesForSitemap,
  getTranslatedArticlesForSitemap,
} from "@/lib/microcms";
import { getAllFilers } from "@/lib/investors";
import { SITE_URL, SITEMAP_IDS, type SitemapId } from "@/lib/site";
import { isIndexableArticle, isIndexableEnArticle, supersededArticleIds } from "@/lib/articleIndexability";
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

// 日付(ISO文字列)の最大値。全エントリに<lastmod>を出すための共通ヘルパー
// （同一形式のISO文字列同士なので文字列比較で新しい方が選べる）。
function maxDate(dates: string[]): string | undefined {
  let latest: string | undefined;
  for (const d of dates) {
    if (d && (!latest || d > latest)) latest = d;
  }
  return latest;
}

// 固定ページ・カテゴリ・FAQ・ランキングなど。<lastmod>は「そのページの内容が実際に変わる
// データ源」から厳密に取る（不正確なlastmodはGoogleに信用されなくなるため）:
// - 記事一覧系ハブ: 表示対象の記事群の最新取引日（日本語ハブ=全記事、/en系=英訳記事）
// - カテゴリ: そのカテゴリの記事の最新取引日
// - EDINET開示系（/activists・/investors）: 最新開示日
// - /about: microCMSのaboutオブジェクトの実updatedAt
// - 静的コンテンツ（/privacy・/en/about・/en/privacy・FAQ）: 更新日を追跡できないため省略
async function pageEntries(): Promise<MetadataRoute.Sitemap> {
  const [articles, translatedArticles, filers, aboutUpdatedAt] = await Promise.all([
    getAllArticlesForSitemap(),
    getTranslatedArticlesForSitemap(),
    getAllFilers(),
    getAboutUpdatedAtForSitemap(),
  ]);
  const latestArticle = maxDate(articles.map((a) => a.dealDate));
  const latestEnArticle = maxDate(translatedArticles.map((a) => a.dealDate));
  const latestDisclosure = maxDate(filers.map((f) => f.latestDiscDate));
  const latestByCategory = latestDealDateBy(articles, (a) => a.dealType);
  const latestByEnCategory = latestDealDateBy(translatedArticles, (a) => a.dealType);
  return [
    { url: SITE_URL, lastModified: latestArticle },
    { url: `${SITE_URL}/en`, lastModified: latestEnArticle },
    { url: `${SITE_URL}/en/investors`, lastModified: latestDisclosure },
    { url: `${SITE_URL}/weekly`, lastModified: latestArticle },
    { url: `${SITE_URL}/activists`, lastModified: latestDisclosure },
    { url: `${SITE_URL}/monthly`, lastModified: latestArticle },
    { url: `${SITE_URL}/trending`, lastModified: latestArticle },
    { url: `${SITE_URL}/about`, lastModified: aboutUpdatedAt },
    { url: `${SITE_URL}/en/about` },
    { url: `${SITE_URL}/privacy` },
    { url: `${SITE_URL}/en/privacy` },
    { url: `${SITE_URL}/faq` },
    // FAQはカテゴリ別ページにQ&A本文を置いているので、各カテゴリもサイトマップに載せる
    // （ハブの/faqからもリンクしているが、確実に拾わせるため）。
    ...FAQ_CATEGORIES.map((category) => ({
      url: `${SITE_URL}/faq/${category.id}`,
    })),
    { url: `${SITE_URL}/investors`, lastModified: latestDisclosure },
    // 3ヶ月リターンランキングは記事ではなくEDINET開示データが源泉。
    { url: `${SITE_URL}/ranking/returns`, lastModified: latestDisclosure },
    { url: `${SITE_URL}/ranking/activist`, lastModified: latestArticle },
    { url: `${SITE_URL}/stocks`, lastModified: latestArticle },
    ...CATEGORIES.map((category) => ({
      url: `${SITE_URL}/category/${encodeURIComponent(category)}`,
      lastModified: latestByCategory.get(category),
    })),
    ...DEAL_TYPES.map((dealType) => ({
      url: `${SITE_URL}/en/category/${DEAL_TYPE_EN[dealType].slug}`,
      lastModified: latestByEnCategory.get(dealType),
    })),
  ];
}

// キー（銘柄コード・取引日など）ごとの記事の最新取引日(=EDINET開示日)。
// 銘柄・日別・月別ページの<lastmod>に使う。
function latestDealDateBy<T extends { dealDate: string }>(
  items: T[],
  keyOf: (item: T) => string
): Map<string, string> {
  const latest = new Map<string, string>();
  for (const item of items) {
    const key = keyOf(item);
    if (!key) continue;
    const prev = latest.get(key);
    if (!prev || item.dealDate > prev) latest.set(key, item.dealDate);
  }
  return latest;
}

async function stockEntries(): Promise<MetadataRoute.Sitemap> {
  const [articles, translatedArticles] = await Promise.all([
    getAllArticlesForSitemap(),
    getTranslatedArticlesForSitemap(),
  ]);
  const latestByStock = latestDealDateBy(articles, (a) => a.stockCode);
  const latestByEnStock = latestDealDateBy(translatedArticles, (a) => a.stockCode);
  return [
    ...[...latestByStock.entries()].map(([code, lastModified]) => ({
      url: `${SITE_URL}/stocks/${code}`,
      lastModified,
    })),
    ...[...latestByEnStock.entries()].map(([code, lastModified]) => ({
      url: `${SITE_URL}/en/stocks/${code}`,
      lastModified,
    })),
  ];
}

async function dateEntries(): Promise<MetadataRoute.Sitemap> {
  const articles = await getAllArticlesForSitemap();
  const latestByDate = latestDealDateBy(articles, (a) => a.dealDate.slice(0, 10));
  const latestByMonth = latestDealDateBy(articles, (a) => a.dealDate.slice(0, 7));
  return [
    ...[...latestByMonth.entries()].map(([month, lastModified]) => ({
      url: `${SITE_URL}/monthly/${month}`,
      lastModified,
    })),
    ...[...latestByDate.entries()].map(([date, lastModified]) => ({
      url: `${SITE_URL}/date/${date}`,
      lastModified,
    })),
  ];
}

// 投資家ページは全2,950件のうち約1,000件が開示1件だけで、保有推移が1点しか出ず
// 中身がほぼ定型になる。全件をsitemapに載せると記事に回るクロール枠を食うため、
// 開示が複数ある投資家（=推移が読める投資家）だけを載せる。
// 除外した投資家のページ自体は残す（noindexにはせず、内部リンクからは辿れる）。
const SITEMAP_MIN_FILER_HOLDINGS = 2;

async function investorEntries(): Promise<MetadataRoute.Sitemap> {
  const filers = await getAllFilers();
  return filers
    .filter((filer) => filer.holdingCount >= SITEMAP_MIN_FILER_HOLDINGS)
    .map((filer) => ({
      url: `${SITE_URL}/investors/${encodeURIComponent(filer.filerName)}`,
      lastModified: filer.latestDiscDate,
    }));
}

// 日英の相互参照(hreflang)は各ページのHTML <head>で宣言済みのため、サイトマップ側には
// 載せない（Googleはどちらか一方の方法で足りる）。changefreq/priorityはGoogleが公式に
// 「無視する」と明言している値なので出さない。全子サイトマップはloc/lastmodの統一形式。
async function articleEntries(): Promise<MetadataRoute.Sitemap> {
  const articles = await getAllArticlesForSitemap();
  // 同一「銘柄×提出者」で最新の1本以外はnoindexにしているため、サイトマップからも外す
  // （載せたままだと「サイトマップに載っているのにnoindex」を送ることになる）。
  const superseded = supersededArticleIds(articles);
  return articles
    .filter((article) => isIndexableArticle(article) && !superseded.has(article.id))
    .map((article) => ({
      url: `${SITE_URL}/articles/${article.id}`,
      lastModified: article.dealDate,
    }));
}

async function enArticleEntries(): Promise<MetadataRoute.Sitemap> {
  // カニバリ判定は全記事（英訳の有無に関わらず）で行う。英訳済みの記事だけでグループを
  // 作ると、記事詳細ページ側（同一銘柄の全記事で判定）と結果がずれる。
  const [translatedArticles, allArticles] = await Promise.all([
    getTranslatedArticlesForSitemap(),
    getAllArticlesForSitemap(),
  ]);
  const superseded = supersededArticleIds(allArticles);
  // 英語版は日本語版より厳しい基準（lib/articleIndexability.tsの注記を参照）。
  return translatedArticles
    .filter((article) => isIndexableEnArticle(article) && !superseded.has(article.id))
    .map((article) => ({
      url: `${SITE_URL}/en/articles/${article.id}`,
      lastModified: article.dealDate,
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
