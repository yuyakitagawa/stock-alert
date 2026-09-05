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

// しきい値は web/publish_blog_articles.py の INDEXABLE_MIN_*（公開済み記事のindex基準）と
// 同じ数値。変更時は両方そろえること。
//
// 新規記事の足切り（同ファイルの MIN_DEAL_AMOUNT_OKU / MIN_RATIO_CHANGE_PT、FAQで
// 読者に公開している数字）は2026-08-29に5億円/1.5ptへ引き上げたが、**こちらは3億円/1.0ptで
// 据え置き**。合わせて上げると既に順位が付いている既存記事の24%をnoindexに落とすことになる。
// 新規記事は必ず足切り≧index基準なので「サイトマップに載っているのにnoindex」は起きない。

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

// ---------------------------------------------------------------------------
// カニバリゼーション対策: 同一「銘柄 × 提出者」で最新の開示1本だけをインデックス対象にする。
// ---------------------------------------------------------------------------
//
// EDINETは同じ提出者が同じ銘柄について変更報告書を何度も出す（実測2026-08-19:
// 直近3か月で5件以上の開示がある提出者が165件、同一銘柄に最大39件の開示）。
// これを全部インデックスさせると「提出者名」「銘柄名＋大株主」という同一クエリに
// 自サイトの記事が何本も並び、内部リンクも評価も分散してどれも順位が上がらない
// （同日のGSC実測: 上位クエリ8件すべて提出者名、掲載順位9.3〜24.5）。
//
// 保有比率は最新の開示が最も正確なので、各グループで最新1本だけを残す。過去の開示は
// /stocks/[code] の「保有比率の推移」と /investors/[filer] から辿れるため情報は失われない
// （ページは残し、noindex,followでリンクだけ生かす）。

export type SupersedableArticle = ArticleIndexabilityInput & {
  id: string;
  stockCode?: string | null;
  filerName?: string | null;
  dealDate?: string | null;
};

// 開示日 → 推定金額 → ID の順で全順序を作る。サイトマップ（全記事）と記事詳細ページ
// （同一銘柄の記事）では入力の並び順が違うため、順序に依存しない比較で勝者を決める。
function comparePreference(a: SupersedableArticle, b: SupersedableArticle): number {
  const dateA = a.dealDate ?? "";
  const dateB = b.dealDate ?? "";
  if (dateA !== dateB) return dateA > dateB ? 1 : -1;
  const amountA = a.dealAmount ?? 0;
  const amountB = b.dealAmount ?? 0;
  if (amountA !== amountB) return amountA > amountB ? 1 : -1;
  return a.id < b.id ? 1 : -1;
}

/**
 * 同一「銘柄 × 提出者」で最新以外の記事IDを返す。
 *
 * 判定対象は isIndexableArticle() を通った記事だけにする。足切り済みの記事を「最新」として
 * 採用すると、グループ内で唯一価値のある記事まで巻き添えでnoindexになる。
 *
 * 呼び出し側はグループの全メンバーを含むリストを渡すこと（記事詳細ページは
 * getArticlesByStockCode()＝同一銘柄・最大100件、サイトマップは全記事）。
 * 同一銘柄の記事が100件を超えない限り両者の判定は一致する。
 */
export function supersededArticleIds(articles: SupersedableArticle[]): Set<string> {
  const winnerByGroup = new Map<string, SupersedableArticle>();
  const superseded = new Set<string>();
  for (const article of articles) {
    if (!article.stockCode || !article.filerName) continue;
    if (!isIndexableArticle(article)) continue;
    const key = `${article.stockCode}\u0000${article.filerName}`;
    const current = winnerByGroup.get(key);
    if (!current) {
      winnerByGroup.set(key, article);
      continue;
    }
    const articleWins = comparePreference(article, current) > 0;
    winnerByGroup.set(key, articleWins ? article : current);
    superseded.add(articleWins ? current.id : article.id);
  }
  return superseded;
}

// ---------------------------------------------------------------------------
// 英語版（en.kujira-watch.com）のインデックス対象判定。日本語版より明確に厳しくする。
// ---------------------------------------------------------------------------
//
// 旧/en時代のGSC実測（2026-08-18・直近3か月）は表示33回・クリック0・平均掲載順位23.3で、
// 需要はゼロではないが極小。英訳を全記事に展開すると日本語版と同じだけのURLがクロール枠を
// 食うので、英語圏の読者が実際に探す「アクティビストの動き」「大型案件」「新規の5%取得」に
// 絞り、それ以外の英訳済み記事はnoindex,follow（ページは残す）にする。
// サブドメインでの再開（2026-09）でも同じ基準を使う。緩める判断はGSCの英語版プロパティの
// 表示回数と blog_crawler_log（host列）の巡回実績を見てから行う。

/** 大型案件とみなす推定取得金額（億円）。 */
export const EN_MIN_DEAL_AMOUNT_OKU = 100;
/** 新規の5%取得とみなす保有比率の変化幅（ポイント）。5%ルールの新規保有は変化幅=保有比率になる。 */
export const EN_NEW_POSITION_RATIO_PT = 5;
/** 新規5%取得でも、規模が小さいものは英語版をindexしない（億円）。 */
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
