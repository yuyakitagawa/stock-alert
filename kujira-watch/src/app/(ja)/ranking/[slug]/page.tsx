import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { formatDate, formatDealAmount } from "@/lib/format";
import { getRecentArticles } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import type { ArticleContent } from "@/types/article";
import AdUnit from "@/components/AdUnit";
import RankingTabNav from "@/components/RankingTabNav";

// ランキングの元データ（記事）は毎時更新なので1時間キャッシュで十分。
export const revalidate = 3600;

// 集計対象期間（暦日）と表示件数。
const RANKING_DAYS = 30;
const RANKING_SIZE = 30;

type RankingSlug = "buys" | "sells" | "filings" | "activist";

const RANKINGS: Record<
  RankingSlug,
  { title: string; description: string; note: string }
> = {
  buys: {
    title: `大口投資家の買い増しランキング（直近${RANKING_DAYS}日）`,
    description:
      `EDINET大量保有報告書で開示された買い増し・新規取得を、推定取得金額が大きい順に並べた` +
      `直近${RANKING_DAYS}日のランキング。大口投資家がいま買っている銘柄が分かります。`,
    note: "推定取得金額（発行済株式数×株価×保有比率の変化幅の概算）が大きい順です。",
  },
  sells: {
    title: `大口投資家の売却ランキング（直近${RANKING_DAYS}日）`,
    description:
      `EDINET大量保有報告書で開示された売却（保有比率の減少）を、推定売却金額が大きい順に並べた` +
      `直近${RANKING_DAYS}日のランキング。大口投資家がいま売っている銘柄が分かります。`,
    note: "推定売却金額（発行済株式数×株価×保有比率の変化幅の概算）が大きい順です。",
  },
  filings: {
    title: `大量保有報告書の件数ランキング（直近${RANKING_DAYS}日）`,
    description:
      `直近${RANKING_DAYS}日にEDINETへ提出された大量保有報告書・変更報告書の件数を銘柄別に集計した` +
      `ランキング。多くの大口投資家が動いている＝注目が集まっている銘柄が分かります。`,
    note: "買い・売り両方向の開示を銘柄ごとに数えた件数順です（同数は合計金額順）。",
  },
  activist: {
    title: `アクティビストが動いた銘柄（直近${RANKING_DAYS}日）`,
    description:
      `直近${RANKING_DAYS}日にアクティビスト（物言う株主）がEDINET大量保有報告書を提出した銘柄の` +
      `一覧。取得も売却も含め、金額規模が大きい順に並べています。`,
    note: "投資家分類が「アクティビスト」の開示のみを金額規模順に並べています。",
  },
};

const SLUGS = Object.keys(RANKINGS) as RankingSlug[];

function isSell(article: ArticleContent): boolean {
  return (article.tags ?? "").split(",").some((tag) => tag.trim() === "売り");
}

type Row = {
  key: string;
  stockCode: string;
  stockName: string;
  amount: number;
  articleId: string;
  articleTitle: string;
  dealDate: string;
  filerName?: string;
  sell: boolean;
  count?: number;
};

function buildRows(slug: RankingSlug, articles: ArticleContent[]): Row[] {
  const toRow = (a: ArticleContent): Row => ({
    key: a.id,
    stockCode: a.stockCode,
    stockName: a.stockName,
    amount: a.dealAmount,
    articleId: a.id,
    articleTitle: a.title,
    dealDate: a.dealDate,
    filerName: a.filerName,
    sell: isSell(a),
  });

  if (slug === "filings") {
    // 銘柄ごとに件数と合計金額を集計し、代表記事は金額が最大の開示にする。
    const byStock = new Map<string, { count: number; total: number; top: ArticleContent }>();
    for (const a of articles) {
      const entry = byStock.get(a.stockCode);
      if (!entry) {
        byStock.set(a.stockCode, { count: 1, total: a.dealAmount, top: a });
      } else {
        entry.count += 1;
        entry.total += a.dealAmount;
        if (a.dealAmount > entry.top.dealAmount) entry.top = a;
      }
    }
    return [...byStock.values()]
      .sort((x, y) => y.count - x.count || y.total - x.total)
      .slice(0, RANKING_SIZE)
      .map(({ count, total, top }) => ({ ...toRow(top), key: top.stockCode, amount: total, count }));
  }

  const filtered =
    slug === "buys"
      ? articles.filter((a) => !isSell(a))
      : slug === "sells"
        ? articles.filter(isSell)
        : articles.filter((a) => a.dealType === "アクティビスト");
  return filtered
    .sort((x, y) => y.dealAmount - x.dealAmount)
    .slice(0, RANKING_SIZE)
    .map(toRow);
}

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
  const rows = buildRows(slug as RankingSlug, contents);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "投資家ランキング", item: `${SITE_URL}/ranking/buys` },
      { "@type": "ListItem", position: 3, name: ranking.title, item: `${SITE_URL}/ranking/${slug}` },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: ranking.title,
    itemListElement: rows.map((row, index) => ({
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
        <Link href="/ranking/buys" className="hover:text-brand-blue">投資家ランキング</Link>
        {" / "}
        <span className="text-foreground/70">{ranking.title}</span>
      </nav>
      {/* タブ切替時に見出しの高さ・位置が変わらないよう、h1は全ランキング共通にして
          個別のランキング名はタブ直下のh2に置く（/ranking・/ranking/trendingと同じ構成）。 */}
      <h1 className="mb-3 text-2xl font-bold text-brand-navy sm:text-3xl">投資家ランキング</h1>
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
      {rows.length === 0 ? (
        <p className="text-foreground/50">直近{RANKING_DAYS}日に該当する開示がありません。</p>
      ) : (
        // /rankingと同じ「1行目=順位＋銘柄名（全幅）／2行目=メタ・金額（右端）」の1件1カード。
        // 銘柄・提出者・解説記事と別々のリンクが3つあるためカード全体はリンクにしない。
        <ul className="card-grid card-grid-wide">
          {rows.map((row, index) => (
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
                {slug === "filings" && row.count !== undefined ? (
                  <span className="font-semibold">{row.count}件の開示</span>
                ) : (
                  <span>{row.sell ? "📉 売却" : "📈 買い増し・新規"}</span>
                )}
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
                  {slug === "filings" && (
                    <span className="kicker ml-1 font-normal text-foreground/40">合計</span>
                  )}
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
