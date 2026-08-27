import type { Metadata } from "next";
import Link from "next/link";
import InfoTip from "@/components/InfoTip";
import { notFound } from "next/navigation";
import { displayFilerName, formatDate, formatDealAmount } from "@/lib/format";
import { getArticleList, getRecentArticles } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { getInvestorReturns, MIN_POSITIONS, RETURN_TRADING_DAYS } from "@/lib/investorReturns";
import { buildStockRows } from "@/lib/rankingStats";
import { getFilerIdMap, investorPath } from "@/lib/investors";
import AdUnit from "@/components/AdUnit";
import InvestorReturnRanking from "@/components/InvestorReturnRanking";
import RelatedArticles from "@/components/RelatedArticles";
import SectorIcon from "@/components/SectorIcon";
import { getCompanyBriefs, type CompanyBrief } from "@/lib/companyInfo";

// activistの元データ（記事）は毎時更新、returnsの元データ（Supabaseのマテリアライズド
// ビュー）は日次更新なので1時間キャッシュで十分。
export const revalidate = 3600;

// activist（記事ベース）の集計対象期間（暦日）と、両ランキング共通の表示件数。
// 期間は/activists・/trendingと同じ7日（直近1週間）。30日窓は毎日ほぼ同じ顔ぶれが並び
// 「直近の動き」が見えないため2026-08-27に短縮した（アクティビストの開示は7日で40件超あり
// 表示上限30件は埋まる）。
const RANKING_DAYS = 7;
const RANKING_SIZE = 30;
// returnsは分類での絞り込みをクライアント側で行うため全件（2026-08-22時点で198名）を渡す。
// ビューはn>=3の投資家しか持たないので件数は増えても数百のオーダーに収まる。
const RANKING_FETCH_LIMIT = 1000;

export type RankingSlug = "returns" | "activist";

// 集計の軸。returnsは投資家別（この投資家が買った銘柄はその後どうなったか）、
// activistはタブに含めない「アクティビストが動いた銘柄」なので銘柄別。
type RankingAxis = "returns" | "stock";

const RANKINGS: Record<
  RankingSlug,
  { axis: RankingAxis; title: string; description: string; detail: string; note: string }
> = {
  returns: {
    axis: "returns",
    title: "3ヶ月リターンランキング（買い開示の成績）",
    description: "投資家が買った銘柄が3ヶ月後にどうなったかを、平均リターン順に並べたランキングです。",
    detail:
      "EDINET大量保有報告書で買い増し・新規取得を開示した投資家について、開示日の終値から" +
      "3ヶ月後（63営業日後）の終値までの騰落率を1件ずつ計算し、平均が高い順に並べています。",
    note:
      `買い開示1件を1ポジションとして等ウェイトで平均した騰落率です（金額加重ではありません）。` +
      `開示${MIN_POSITIONS}件以上・3ヶ月が経過した開示のみが対象。`,
  },
  activist: {
    axis: "stock",
    title: `アクティビストが動いた銘柄（直近${RANKING_DAYS}日）`,
    description: `直近${RANKING_DAYS}日にアクティビスト（物言う株主）が動いた銘柄を金額規模順に並べています。`,
    detail: "EDINET大量保有報告書の提出をもとに、取得も売却も含めて集計しています。",
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

  const returnRows =
    ranking.axis === "returns" ? await getInvestorReturns(RANKING_FETCH_LIMIT) : [];
  const recentArticles =
    ranking.axis === "stock" ? (await getRecentArticles(RANKING_DAYS)).contents : [];
  const stockRows = ranking.axis === "stock" ? buildStockRows(recentArticles, RANKING_SIZE) : [];
  const filerIds = ranking.axis === "stock" ? await getFilerIdMap() : {};
  // 銘柄カードの業種アイコン用（会社ロゴは持てないため業種で代替）。
  const stockBriefs =
    ranking.axis === "stock"
      ? await getCompanyBriefs(stockRows.map((row) => row.stockCode)).catch(
          () => new Map<string, CompanyBrief>()
        )
      : new Map<string, CompanyBrief>();
  const rowCount = ranking.axis === "returns" ? returnRows.length : stockRows.length;

  // ランキングの文脈に合ったアイキャッチ付き記事カード。returnsはランキング上位投資家の
  // 直近記事（新着順・1投資家1件まで）、activistは既に取得済みの直近7日の記事から
  // 金額規模の大きい取引（1銘柄1件まで）。取れなくてもランキング自体は成立させる。
  let relatedArticles: typeof recentArticles = [];
  if (ranking.axis === "returns") {
    const topFilers = new Set(returnRows.slice(0, RANKING_SIZE).map((row) => row.filerName));
    const { contents: latest } = await getArticleList({ limit: 50 }).catch(() => ({ contents: [] }));
    const seenFilers = new Set<string>();
    relatedArticles = latest
      .filter((article) => {
        if (!article.filerName || !topFilers.has(article.filerName) || seenFilers.has(article.filerName)) {
          return false;
        }
        seenFilers.add(article.filerName);
        return true;
      })
      .slice(0, 4);
  } else {
    const seenCodes = new Set<string>();
    relatedArticles = [...recentArticles]
      .sort((a, b) => b.dealAmount - a.dealAmount)
      .filter((article) => {
        if (seenCodes.has(article.stockCode)) return false;
        seenCodes.add(article.stockCode);
        return true;
      })
      .slice(0, 4);
  }

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
        ? // 構造化データは初期表示（絞り込みなしの上位30名）に揃える。
          returnRows.slice(0, RANKING_SIZE).map((row, index) => ({
            "@type": "ListItem",
            position: index + 1,
            name: displayFilerName(row.filerName),
            url: `${SITE_URL}${investorPath(row.filerId, row.filerName)}`,
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
      <p className="mb-6 text-sm text-foreground/50">
        {ranking.description}
        <InfoTip
          content={`${ranking.detail} ${ranking.note} 出典はEDINET大量保有報告書。過去の成績であり、将来の値動きや投資助言を示すものではありません。`}
        />
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
        <>
          <InvestorReturnRanking rows={returnRows} size={RANKING_SIZE} />
          <p className="mt-2 text-xs leading-relaxed text-foreground/40">
            集計対象は買い開示{MIN_POSITIONS}件以上の投資家です。リターンは開示日（休場なら直後の営業日）の終値を基準に
            {RETURN_TRADING_DAYS}営業日後の終値までを計算したもので、実際の取得単価・売却時期は反映していません。
            日経平均比は同じ期間の日経平均の騰落率との差（％pt）です。
          </p>
        </>
      ) : (
        <ul className="card-grid card-grid-wide">
          {stockRows.map((row, index) => (
            <li key={row.key} className="card">
              <span className="flex items-start gap-2">
                <span className="w-5 shrink-0 font-bold tabular-nums text-foreground/40">
                  {index + 1}
                </span>
                <SectorIcon sector={stockBriefs.get(row.stockCode)?.sector} />
                <Link
                  href={`/stocks/${row.stockCode}`}
                  className="min-w-0 grow font-medium text-brand-blue [overflow-wrap:anywhere] hover:underline"
                >
                  {row.stockName}（{row.stockCode}）
                </Link>
              </span>
              <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-foreground/60">
                <span>{row.sell ? "📉 売却" : "📈 買い増し・新規"}</span>
                {row.filerName && (
                  <Link
                    href={investorPath(filerIds[row.filerName], row.filerName)}
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
      <div className="mt-10">
        <RelatedArticles
          title={ranking.axis === "returns" ? "上位投資家の最新記事" : "アクティビスト取引の解説記事"}
          lead={
            ranking.axis === "returns"
              ? "リターンランキング上位の投資家による、直近の取引の解説記事です。"
              : `直近${RANKING_DAYS}日の取引から推定金額の大きいものをピックアップ。`
          }
          articles={relatedArticles}
        />
      </div>
      <AdUnit placement="bottom" />
    </div>
  );
}
