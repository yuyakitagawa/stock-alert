import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { displayFilerName, formatDate, formatDealAmount } from "@/lib/format";
import { getRecentArticles } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import {
  buildFilerRows,
  buildStockRows,
  type RankingSlug,
} from "@/lib/rankingStats";
import AdUnit from "@/components/AdUnit";
import DealTypeLabel from "@/components/DealTypeLabel";
import RankingTabNav from "@/components/RankingTabNav";

// ランキングの元データ（記事）は毎時更新なので1時間キャッシュで十分。
export const revalidate = 3600;

// 集計対象期間（暦日）と表示件数。
const RANKING_DAYS = 30;
const RANKING_SIZE = 30;

// 集計の軸。buys/sellsは「月間ランキング」タブの一員なので投資家別に集計する
// （銘柄別の同種ランキングは/trendingや各銘柄ページの役割）。activistはタブに含めない
// 「アクティビストが動いた銘柄」なので銘柄別のまま。
type RankingAxis = "filer" | "stock";

const RANKINGS: Record<
  RankingSlug,
  { axis: RankingAxis; title: string; description: string; note: string }
> = {
  buys: {
    axis: "filer",
    title: `買い増しランキング（直近${RANKING_DAYS}日）`,
    description:
      `EDINET大量保有報告書で開示された買い増し・新規取得を投資家別に合計し、推定取得金額が` +
      `大きい順に並べた直近${RANKING_DAYS}日のランキング。いちばん買っている大口投資家が分かります。`,
    note: "投資家ごとの推定取得金額（発行済株式数×株価×保有比率の変化幅の概算）の合計が大きい順です。",
  },
  sells: {
    axis: "filer",
    title: `売却ランキング（直近${RANKING_DAYS}日）`,
    description:
      `EDINET大量保有報告書で開示された売却（保有比率の減少）を投資家別に合計し、推定売却金額が` +
      `大きい順に並べた直近${RANKING_DAYS}日のランキング。いちばん売っている大口投資家が分かります。`,
    note: "投資家ごとの推定売却金額（発行済株式数×株価×保有比率の変化幅の概算）の合計が大きい順です。",
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

  const { contents } = await getRecentArticles(RANKING_DAYS);
  const filerRows =
    ranking.axis === "filer" ? buildFilerRows(slug as "buys" | "sells", contents, RANKING_SIZE) : [];
  const stockRows = ranking.axis === "stock" ? buildStockRows(contents, RANKING_SIZE) : [];
  const rowCount = ranking.axis === "filer" ? filerRows.length : stockRows.length;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "月間ランキング", item: `${SITE_URL}/ranking` },
      { "@type": "ListItem", position: 3, name: ranking.title, item: `${SITE_URL}/ranking/${slug}` },
    ],
  };

  // 構造化データの並びも表示と同じ軸にする（投資家別ランキングは投資家ページを指す）。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: ranking.title,
    itemListElement:
      ranking.axis === "filer"
        ? filerRows.map((row, index) => ({
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
        <Link href="/ranking" className="hover:text-brand-blue">月間ランキング</Link>
        {" / "}
        <span className="text-foreground/70">{ranking.title}</span>
      </nav>
      {/* タブ切替時に見出しの高さ・位置が変わらないよう、h1は全ランキング共通にして
          個別のランキング名はタブ直下のh2に置く（/ranking・/ranking/trendingと同じ構成）。 */}
      <h1 className="mb-3 text-2xl font-bold text-brand-navy sm:text-3xl">月間ランキング</h1>
      <RankingTabNav current={`/ranking/${slug}`} />
      <h2 className="mb-1 text-xl font-bold text-brand-navy">{ranking.title}</h2>
      <p className="mb-2 text-sm text-foreground/50">{ranking.description}</p>
      <p className="mb-6 text-xs text-foreground/40">
        ※{ranking.note} 出典はEDINET大量保有報告書。投資助言ではありません。
      </p>
      {/* アクティビストランキングだけは/activists（アクティビストの動き）と対の関係にあるため
          相互リンクを置く。他のタブページには無関係なリンクを並べない。 */}
      {slug === "activist" && (
        <nav aria-label="関連ページ" className="mb-6 flex flex-wrap gap-x-4 gap-y-1 text-sm">
          <Link href="/activists" className="text-brand-blue hover:underline">
            アクティビストの動き
          </Link>
        </nav>
      )}
      {rowCount === 0 ? (
        <p className="text-foreground/50">直近{RANKING_DAYS}日に該当する開示がありません。</p>
      ) : ranking.axis === "filer" ? (
        // /rankingと同じ「1行目=順位＋投資家名（全幅）／2行目=メタ・金額（右端）」の1件1カード。
        // 投資家・代表銘柄・解説記事と別々のリンクが3つあるためカード全体はリンクにしない。
        <>
          <p className="kicker mb-2 text-foreground/40">
            順位・投資家 ／ 分類・開示件数・代表銘柄・推定金額の合計
          </p>
          <ul className="card-grid card-grid-wide">
            {filerRows.map((row, index) => (
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
                  {row.category && <DealTypeLabel dealType={row.category} />}
                  <span className="font-semibold">{row.count}件の開示</span>
                  <Link
                    href={`/stocks/${row.topStockCode}`}
                    className="text-brand-blue hover:underline"
                  >
                    {row.topStockName}（{row.topStockCode}）
                  </Link>
                  <Link href={`/articles/${row.topArticleId}`} className="text-brand-blue hover:underline">
                    解説記事
                  </Link>
                  <span>最終開示{formatDate(row.latestDealDate)}</span>
                  <span className="ml-auto whitespace-nowrap text-sm font-semibold tabular-nums text-brand-navy">
                    {formatDealAmount(row.amount)}
                    <span className="kicker ml-1 font-normal text-foreground/40">合計</span>
                  </span>
                </span>
              </li>
            ))}
          </ul>
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
