import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { displayFilerName, formatDate, formatDealAmount } from "@/lib/format";
import { getRecentArticles } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import {
  getInvestorReturns,
  getInvestorReturnsSummary,
  formatSignedPercent,
  formatSignedPoint,
  MIN_POSITIONS,
  RETURN_TRADING_DAYS,
} from "@/lib/investorReturns";
import { buildStockRows } from "@/lib/rankingStats";
import AdUnit from "@/components/AdUnit";
import DealTypeLabel from "@/components/DealTypeLabel";

// activistの元データ（記事）は毎時更新、returnsの元データ（Supabaseのマテリアライズド
// ビュー）は日次更新なので1時間キャッシュで十分。
export const revalidate = 3600;

// activist（記事ベース）の集計対象期間（暦日）と、両ランキング共通の表示件数。
const RANKING_DAYS = 30;
const RANKING_SIZE = 30;

export type RankingSlug = "returns" | "activist";

// 集計の軸。returnsは投資家別（この投資家が買った銘柄はその後どうなったか）、
// activistはタブに含めない「アクティビストが動いた銘柄」なので銘柄別。
type RankingAxis = "returns" | "stock";

const RANKINGS: Record<
  RankingSlug,
  { axis: RankingAxis; title: string; description: string; note: string }
> = {
  returns: {
    axis: "returns",
    title: "3ヶ月リターンランキング（買い開示の成績）",
    description:
      "EDINET大量保有報告書で買い増し・新規取得を開示した投資家について、開示日の終値から" +
      "3ヶ月後（63営業日後）の終値までの騰落率を1件ずつ計算し、平均が高い順に並べた" +
      "ランキング。「この投資家が買った銘柄はその後どうなったか」が分かります。",
    note:
      `買い開示1件を1ポジションとして等ウェイトで平均した騰落率です（金額加重ではありません）。` +
      `開示${MIN_POSITIONS}件以上・3ヶ月が経過した開示のみが対象。`,
  },
  activist: {
    axis: "stock",
    title: `アクティビストが動いた銘柄（直近${RANKING_DAYS}日）`,
    description:
      `直近${RANKING_DAYS}日にアクティビスト（物言う株主）がEDINET大量保有報告書を提出した銘柄の` +
      `一覧。取得も売却も含め、金額規模が大きい順に並べています。`,
    note: "投資家分類が「アクティビスト」の開示のみを金額規模順に並べています。",
  },
};

const SLUGS = Object.keys(RANKINGS) as RankingSlug[];

type Props = { params: Promise<{ slug: string }> };

export function generateStaticParams() {
  return SLUGS.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const ranking = RANKINGS[slug as RankingSlug];
  if (!ranking) return {};
  const url = `${SITE_URL}/ranking/${slug}`;
  return {
    title: ranking.title,
    description: ranking.description,
    alternates: { canonical: url },
    openGraph: { title: ranking.title, description: ranking.description, url },
  };
}

export default async function RankingSlugPage({ params }: Props) {
  const { slug } = await params;
  const ranking = RANKINGS[slug as RankingSlug];
  if (!ranking) notFound();

  const returnRows = ranking.axis === "returns" ? await getInvestorReturns(RANKING_SIZE) : [];
  const returnSummary =
    ranking.axis === "returns" ? await getInvestorReturnsSummary() : { filerCount: 0, latestBuyDate: null };
  // 記事(microCMS)を読むのはactivistだけ。returnsの集計元はSupabaseなので取りに行かない。
  const stockRows =
    ranking.axis === "stock"
      ? buildStockRows((await getRecentArticles(RANKING_DAYS)).contents, RANKING_SIZE)
      : [];
  const rowCount = ranking.axis === "returns" ? returnRows.length : stockRows.length;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "投資家ランキング", item: `${SITE_URL}/ranking/returns` },
      { "@type": "ListItem", position: 3, name: ranking.title, item: `${SITE_URL}/ranking/${slug}` },
    ],
  };

  // 構造化データの並びも表示と同じ軸にする（投資家別ランキングは投資家ページを指す）。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: ranking.title,
    itemListElement:
      ranking.axis === "returns"
        ? returnRows.map((row, index) => ({
            "@type": "ListItem",
            position: index + 1,
            name: displayFilerName(row.filerName),
            url: `${SITE_URL}/investors/${encodeURIComponent(row.filerName)}`,
          }))
        : stockRows.map((row, index) => ({
            "@type": "ListItem",
            position: index + 1,
            name: `${row.stockName}（${row.stockCode}）`,
            url: `${SITE_URL}/stocks/${row.stockCode}`,
          })),
  };

  return (
    <div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href="/ranking/returns" className="hover:text-brand-blue">投資家ランキング</Link>
        {" / "}
        <span className="text-foreground/70">{ranking.title}</span>
      </nav>
      {/* h1はヘッダー・フッターのラベル（投資家ランキング）と揃え、個別のランキング名は
          h2に置く。2026-08-21に開示急増投資家ランキングを廃止してタブは無くなったが、
          パンくず・ナビの語との一致を優先してこの2段構成のままにしている。 */}
      <h1 className="mb-3 text-2xl font-bold text-brand-navy sm:text-3xl">投資家ランキング</h1>
      <h2 className="mb-1 text-xl font-bold text-brand-navy">{ranking.title}</h2>
      <p className="mb-2 text-sm text-foreground/50">{ranking.description}</p>
      <p className="mb-6 text-xs text-foreground/40">
        ※{ranking.note} 出典はEDINET大量保有報告書。過去の成績であり、将来の値動きや投資助言を示すものではありません。
      </p>
      {/* アクティビストランキングだけは/activists（アクティビスト注目銘柄）と対の関係にあるため
          相互リンクを置く。他のタブページには無関係なリンクを並べない。 */}
      {slug === "activist" && (
        <nav aria-label="関連ページ" className="mb-6 flex flex-wrap gap-x-4 gap-y-1 text-sm">
          <Link href="/activists" className="text-brand-blue hover:underline">
            アクティビスト注目銘柄
          </Link>
        </nav>
      )}
      {rowCount === 0 ? (
        <p className="text-foreground/50">
          {ranking.axis === "returns"
            ? "集計対象の開示がまだありません。"
            : `直近${RANKING_DAYS}日に該当する開示がありません。`}
        </p>
      ) : ranking.axis === "returns" ? (
        // 「1行目=順位＋投資家名（全幅）／2行目=メタ・平均リターン（右端）」の1件1カード。
        // 投資家・最高銘柄と別々のリンクが2つあるためカード全体はリンクにしない。
        <>
          <p className="kicker mb-2 text-foreground/40">
            順位・投資家 ／ 分類・買い開示件数・勝率・日経平均比・最も上がった銘柄・3ヶ月平均リターン
          </p>
          <ul className="card-grid card-grid-wide">
            {returnRows.map((row, index) => (
              <li key={row.filerName} className="card">
                <span className="flex items-baseline gap-2">
                  <span className="w-5 shrink-0 font-bold tabular-nums text-foreground/40">
                    {index + 1}
                  </span>
                  <Link
                    href={`/investors/${encodeURIComponent(row.filerName)}`}
                    className="min-w-0 grow font-medium text-brand-blue [overflow-wrap:anywhere] hover:underline"
                  >
                    {displayFilerName(row.filerName)}
                  </Link>
                </span>
                <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 pl-7 text-xs text-foreground/60">
                  <DealTypeLabel dealType={row.category} />
                  <span className="font-semibold">買い開示{row.positionCount}件</span>
                  <span>勝率{row.winRate}%</span>
                  {row.avgExcessReturn !== null && (
                    <span>日経平均比{formatSignedPoint(row.avgExcessReturn)}</span>
                  )}
                  <span className="flex items-center gap-1">
                    最高
                    <Link
                      href={`/stocks/${row.bestStockCode}`}
                      className="text-brand-blue hover:underline"
                    >
                      {row.bestStockName}（{formatSignedPercent(row.bestReturn, 0)}）
                    </Link>
                  </span>
                  <span>最終買い{formatDate(row.latestBuyDate)}</span>
                  <span
                    className={`ml-auto whitespace-nowrap text-sm font-semibold tabular-nums ${
                      row.avgReturn >= 0 ? "text-gain" : "text-loss"
                    }`}
                  >
                    {formatSignedPercent(row.avgReturn)}
                    <span className="kicker ml-1 font-normal text-foreground/40">3ヶ月平均</span>
                  </span>
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs leading-relaxed text-foreground/40">
            集計対象は開示{MIN_POSITIONS}件以上の投資家{returnSummary.filerCount}名で、上位
            {returnRows.length}名を表示しています。リターンは開示日（休場なら直後の営業日）の終値を基準に
            {RETURN_TRADING_DAYS}営業日後の終値までを計算したもので、実際の取得単価・売却時期は反映していません。
            日経平均比は同じ期間の日経平均の騰落率との差（％pt）です。
          </p>
        </>
      ) : (
        <ul className="card-grid card-grid-wide">
          {stockRows.map((row, index) => (
            <li key={row.key} className="card">
              <span className="flex items-baseline gap-2">
                <span className="w-5 shrink-0 font-bold tabular-nums text-foreground/40">
                  {index + 1}
                </span>
                <Link
                  href={`/stocks/${row.stockCode}`}
                  className="min-w-0 grow font-medium text-brand-blue [overflow-wrap:anywhere] hover:underline"
                >
                  {row.stockName}（{row.stockCode}）
                </Link>
              </span>
              <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 pl-7 text-xs text-foreground/60">
                <span>{row.sell ? "📉 売却" : "📈 買い増し・新規"}</span>
                {row.filerName && (
                  <Link
                    href={`/investors/${encodeURIComponent(row.filerName)}`}
                    className="text-brand-blue hover:underline"
                  >
                    {row.filerName}
                  </Link>
                )}
                <span>{formatDate(row.dealDate)}</span>
                <Link href={`/articles/${row.articleId}`} className="text-brand-blue hover:underline">
                  解説記事
                </Link>
                <span className="ml-auto whitespace-nowrap text-sm font-semibold tabular-nums text-brand-navy">
                  {formatDealAmount(row.amount)}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}
