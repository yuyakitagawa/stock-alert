// 自動生成の集約ページ（取引日別・投資家別・銘柄別）をインデックス対象にするかの判定。
//
// 2026-08-24、AdSenseの審査に「有用性の低いコンテンツ」で不承認。実測すると全4,756URLのうち
// 75%がこれらの自動生成ページで、中身は次のように薄かった:
//   - 取引日別 223件: 記事1件だけが87件、2件以下が149件（本文の中央値303字）
//   - 投資家別 2,972件: 解説文が無いものが2,152件、開示1件だけが1,002件
//   - 銘柄別 664件: 記事1件だけが482件
// Googleの「質の低いコンテンツ」ガイドラインが挙げる cookie cutter pages そのものなので、
// 中身が一定量あるページだけを公開する。
//
// 2026-08-29変更: 当初はこれらを noindex,follow にしてページ自体は残していたが、noindexは
// 「評価しないでくれ」という指定でしかなくURLは存在し続ける（実測で約4,969URLが残っていた）。
// オーナー判断で、条件を満たさないページは 404 にする＝公開しないことにした。
// 公開判定と内部リンクの出し分けは lib/publishedPages.ts に置いてある。
//
// 判定は必ずこのファイルに集約すること。ページ側・サイトマップ側・内部リンク側で条件がずれると
// 「サイトマップに載っているのに404」「リンクを踏むと404」という矛盾をGoogleと読者に送ることになる。

/** 取引日別ページ: この件数以上の記事がある日だけインデックスする。 */
export const INDEXABLE_MIN_DATE_ARTICLES = 3;

/** 投資家別ページ: この件数以上の開示がある投資家だけ公開する（保有比率の「推移」が読める下限）。 */
export const INDEXABLE_MIN_FILER_HOLDINGS = 2;

/**
 * 投資家別ページ: この件数以上の開示がある投資家だけを検索インデックス対象にする
 * （サイトマップに載せる／noindexを付けない）。公開の下限（上記）より厳しい。
 *
 * 2026-09-21の実測。9/15以降にGooglebotが取得したURLの53%（706回・424URL）が投資家ページで、
 * 一方で8/31以降の新記事249件は1件も取得されていなかった（GSCでは全件「検出-インデックス未登録」）。
 * 巡回の割り当てが、成果の出ていない集約ページに寄っている。直近28日のGSC実績は
 *   開示2〜3件: 167ページ → クリック0・表示24
 *   開示4〜9件: 172ページ → クリック0・表示53
 *   開示10件〜: 148ページ → クリック2・表示327
 * で、10件未満の339ページはクリック0だった。ここをサイトマップから外しnoindexにして、
 * 巡回を記事へ寄せる。8月の大量404で検索に出ていたURLを失った反省から、ページ自体は
 * 404にせず200のまま残す（内部リンクもそのまま＝リンク切れを作らない）。
 */
export const SEARCH_INDEXABLE_MIN_FILER_HOLDINGS = 10;

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
 * 投資家別 /investors/[filer] を検索インデックス対象にするか。
 * falseでも公開（200）は続けるので、ページ側は notFound() ではなく noindex,follow を付ける。
 * 「公開するが検索には出さない」ページはサイトマップにも載せない
 * （載せたままだと「サイトマップに載っているのにnoindex」を送ることになる）。
 */
export function isSearchIndexableInvestorPage(input: {
  holdingCount: number;
  hasProfile: boolean;
}): boolean {
  return (
    isIndexableInvestorPage(input) &&
    input.holdingCount >= SEARCH_INDEXABLE_MIN_FILER_HOLDINGS
  );
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
