// 自動生成の集約ページ（取引日別・投資家別・銘柄別）をインデックス対象にするかの判定。
//
// 2026-08-24、AdSenseの審査に「有用性の低いコンテンツ」で不承認。実測すると全4,756URLのうち
// 75%がこれらの自動生成ページで、中身は次のように薄かった:
//   - 取引日別 223件: 記事1件だけが87件、2件以下が149件（本文の中央値303字）
//   - 投資家別 2,972件: 解説文が無いものが2,152件、開示1件だけが1,002件
//   - 銘柄別 664件: 記事1件だけが482件
// Googleの「質の低いコンテンツ」ガイドラインが挙げる cookie cutter pages そのものなので、
// 中身が一定量あるページだけをインデックス対象にし、残りは noindex,follow にする。
// ページ自体は消さない（内部リンクからは辿れるので、読者にとっての回遊性は落とさない）。
//
// 判定は必ずこのファイルに集約すること。ページ側とサイトマップ側で条件がずれると
// 「サイトマップに載っているのにnoindex」という矛盾したシグナルをGoogleに送ることになる。

/** 取引日別ページ: この件数以上の記事がある日だけインデックスする。 */
export const INDEXABLE_MIN_DATE_ARTICLES = 3;

/** 投資家別ページ: この件数以上の開示がある投資家だけインデックスする（保有比率の「推移」が読める下限）。 */
export const INDEXABLE_MIN_FILER_HOLDINGS = 2;

/** 銘柄別ページ: 解説記事がこの件数以上あればインデックスする（事業内容の説明が無くても可）。 */
export const INDEXABLE_MIN_STOCK_ARTICLES = 2;

/**
 * 取引日別 /date/[date]。
 * 開示が1〜2件しかない日は、記事へのリンクが数本並ぶだけで記事本文と内容が重複する。
 */
export function isIndexableDatePage(articleCount: number): boolean {
  return articleCount >= INDEXABLE_MIN_DATE_ARTICLES;
}

/**
 * 投資家別 /investors/[filer]。
 * 開示が複数あって推移が読めること、かつ投資家の解説文があることを条件にする。
 * 解説文の無い開示1件だけのページは、EDINETの1行をそのまま表に起こしただけになる。
 */
export function isIndexableInvestorPage(input: {
  holdingCount: number;
  hasProfile: boolean;
}): boolean {
  return input.holdingCount >= INDEXABLE_MIN_FILER_HOLDINGS && input.hasProfile;
}

/**
 * 銘柄別 /stocks/[code]。
 * 解説記事が複数あるか、1件でもその会社が何をしているかの説明が載っていることを条件にする。
 * 記事0件のページ（開示テーブルと株価だけ）は従来どおり対象外。
 */
export function isIndexableStockPage(input: {
  articleCount: number;
  hasCompanyDescription: boolean;
}): boolean {
  if (input.articleCount >= INDEXABLE_MIN_STOCK_ARTICLES) return true;
  return input.articleCount >= 1 && input.hasCompanyDescription;
}
