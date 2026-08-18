// 記事をインデックス対象（sitemap掲載＋index許可）にするかの判定。
//
// EDINETの開示には「保有比率0.04%・推定取得額0億円」のような実質的にニュース価値の無い
// 変更報告書が大量に含まれる。これをそのまま記事化するとGoogleに /articles/ テンプレート
// 全体を低品質と判断され、新規記事がクロールすらされなくなる（2026-08-18時点で
// GSC「検出 - インデックス未登録」が多発した主因）。金額か保有比率の変化のどちらかが
// 基準を超える開示だけをインデックス対象にし、それ以外はnoindex＋sitemap除外にする。
//
// 判定はmicroCMSのフィールド（dealAmount / ratioChangePct）だけで完結させること。
// EDINET(Supabase)の値を混ぜるとsitemapと記事ページで判定がずれ、
// 「サイトマップに載っているのにnoindex」という自己矛盾をGoogleに送ることになる。

/** 推定取得金額の下限（億円）。これ以上なら比率変化が小さくてもインデックス対象。 */
export const INDEXABLE_MIN_DEAL_AMOUNT_OKU = 3;
/** 保有比率の変化幅の下限（ポイント）。金額が小さくても保有方針の変化は価値がある。 */
export const INDEXABLE_MIN_RATIO_CHANGE_PT = 1;

export type ArticleIndexabilityInput = {
  dealAmount?: number | null;
  ratioChangePct?: number | null;
};

export function isIndexableArticle(article: ArticleIndexabilityInput): boolean {
  if ((article.dealAmount ?? 0) >= INDEXABLE_MIN_DEAL_AMOUNT_OKU) return true;
  return Math.abs(article.ratioChangePct ?? 0) >= INDEXABLE_MIN_RATIO_CHANGE_PT;
}
