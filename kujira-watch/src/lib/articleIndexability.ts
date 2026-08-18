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

// 英語版(/en)の対象判定。日本語版より明確に厳しくする。
//
// GSCの実測（2026-08-18・直近3か月・ページ`/en/`）は 表示33回・クリック0・平均掲載順位23.3で、
// 需要はゼロではないが極小。一方で英訳は全911記事に自動展開されており、日本語版と同じだけの
// URLがクロール枠を食っていた。英語圏の読者が実際に探すのは「アクティビストの動き」
// 「大型案件」「新規の5%取得」なので、その3つに絞って残りは英訳しない（既存分はnoindex）。
// この基準を緩める/広げる判断は、GSCの`/en/`の表示回数の推移を見てから行う。

/** 大型案件とみなす推定取得金額（億円）。 */
export const EN_MIN_DEAL_AMOUNT_OKU = 100;
/** 新規の5%取得とみなす保有比率の変化幅（ポイント）。5%ルールの新規保有は変化幅=保有比率になる。 */
export const EN_NEW_POSITION_RATIO_PT = 5;
/** 新規5%取得でも、規模が小さいものは英語版を作らない（億円）。 */
export const EN_NEW_POSITION_MIN_AMOUNT_OKU = 20;
/** 英語圏の関心が最も高い投資家分類。 */
export const EN_ALWAYS_DEAL_TYPE = "アクティビスト";

export type EnArticleIndexabilityInput = ArticleIndexabilityInput & {
  dealType?: string | string[] | null;
};

export function isIndexableEnArticle(article: EnArticleIndexabilityInput): boolean {
  const dealTypes = Array.isArray(article.dealType)
    ? article.dealType
    : article.dealType
      ? [article.dealType]
      : [];
  if (dealTypes.includes(EN_ALWAYS_DEAL_TYPE)) return true;
  const amount = article.dealAmount ?? 0;
  if (amount >= EN_MIN_DEAL_AMOUNT_OKU) return true;
  const ratioChange = Math.abs(article.ratioChangePct ?? 0);
  return ratioChange >= EN_NEW_POSITION_RATIO_PT && amount >= EN_NEW_POSITION_MIN_AMOUNT_OKU;
}
