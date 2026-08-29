// 自動生成の集約ページ（銘柄・投資家・取引日）を「公開するか」の判定。
//
// 2026-08-24のAdSense不承認（有用性の低いコンテンツ）を受けて、中身の薄い集約ページは
// いったん noindex にした。ただし noindex は「評価しないでくれ」という指定でしかなく、
// URL自体は存在し続ける（実測2026-08-29: sitemap 2,252に対し、noindexのまま残っている
// 集約ページが銘柄2,355・投資家2,417・取引日197＝約4,969URLあった）。
// 2026-08-29、オーナー判断でこれらは noindex ではなく 404 にする＝公開しないことにした。
//
// **重要: ページを404にする以上、内部リンクも必ず同じ判定で出し分けること。**
// リンクだけ残すと記事本文・一覧・ランキングから大量のリンク切れが出て、
// 「薄いページ」を消した代わりに「壊れたサイト」をGoogleに見せることになる。
// そのため href は必ず stockHref()/dateHref()/investorHref() で組み立て、
// null が返ったらリンクにせず素のテキストで出す（呼び出し側の分岐を1行で済ませるため
// 戻り値を string | null にしてある）。
//
// 判定の条件そのものは lib/pageIndexability.ts に置いたままにする（サイトマップと共通）。
import { getAllArticlesForSitemap } from "@/lib/microcms";
import { getStockDescriptionCodes } from "@/lib/companyInfo";
import { getAllFilers, getFilersWithProfile } from "@/lib/investors";
import { investorPath } from "@/lib/investorPath";
import {
  isIndexableDatePage,
  isIndexableInvestorPage,
  isIndexableStockPage,
} from "@/lib/pageIndexability";

// 集合はキャッシュに入れない（unstable_cacheは戻り値をJSONで直列化するためSetが{}になる。
// 2026-08-25にこれで本番ビルドが24時間落ちた）。元になる配列側が各モジュールで
// unstable_cacheされているので、ここでの組み立ては毎回やっても軽い。

/** 公開する銘柄ページのコード集合。 */
export async function getPublishedStockCodes(): Promise<Set<string>> {
  const [articles, describedCodes] = await Promise.all([
    getAllArticlesForSitemap(),
    getStockDescriptionCodes(),
  ]);
  const countByStock = new Map<string, number>();
  for (const article of articles) {
    if (!article.stockCode) continue;
    countByStock.set(article.stockCode, (countByStock.get(article.stockCode) ?? 0) + 1);
  }
  const published = new Set<string>();
  for (const [code, articleCount] of countByStock) {
    if (isIndexableStockPage({ articleCount, hasCompanyDescription: describedCodes.has(code) })) {
      published.add(code);
    }
  }
  return published;
}

/** 公開する取引日ページの日付集合（YYYY-MM-DD）。 */
export async function getPublishedDates(): Promise<Set<string>> {
  const articles = await getAllArticlesForSitemap();
  const countByDate = new Map<string, number>();
  for (const article of articles) {
    const date = article.dealDate?.slice(0, 10);
    if (!date) continue;
    countByDate.set(date, (countByDate.get(date) ?? 0) + 1);
  }
  const published = new Set<string>();
  for (const [date, count] of countByDate) {
    if (isIndexableDatePage(count)) published.add(date);
  }
  return published;
}

/** 公開する投資家ページの提出者名集合。 */
export async function getPublishedFilerNames(): Promise<Set<string>> {
  const [filers, filersWithProfile] = await Promise.all([
    getAllFilers(),
    getFilersWithProfile(),
  ]);
  const published = new Set<string>();
  for (const filer of filers) {
    if (
      isIndexableInvestorPage({
        holdingCount: filer.holdingCount,
        hasProfile: filersWithProfile.has(filer.filerName),
      })
    ) {
      published.add(filer.filerName);
    }
  }
  return published;
}

/** 銘柄ページへのリンク先。公開していない銘柄はnull（呼び出し側はテキストで出す）。 */
export async function stockHref(code: string | null | undefined): Promise<string | null> {
  if (!code) return null;
  return (await getPublishedStockCodes()).has(code) ? `/stocks/${code}` : null;
}

/** 取引日ページへのリンク先。公開していない日はnull。 */
export async function dateHref(date: string | null | undefined): Promise<string | null> {
  if (!date) return null;
  const day = date.slice(0, 10);
  return (await getPublishedDates()).has(day) ? `/date/${day}` : null;
}

/** 投資家ページへのリンク先。公開していない提出者はnull。 */
export async function investorHref(
  filerId: number | null | undefined,
  filerName: string | null | undefined
): Promise<string | null> {
  if (!filerName) return null;
  return (await getPublishedFilerNames()).has(filerName)
    ? investorPath(filerId, filerName)
    : null;
}
