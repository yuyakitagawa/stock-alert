import type { Metadata } from "next";
import Link from "next/link";
import { formatDate, formatMonth } from "@/lib/format";
import { getAllStocksForIndex } from "@/lib/microcms";
import { getHoldingsInRange } from "@/lib/investors";
import { getMonthlyDisclosureCounts } from "@/lib/disclosures";
import DisclosureTrendChart from "@/components/DisclosureTrendChart";
import TrendingTable from "@/components/TrendingTable";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { buildTrendingIssuers } from "@/lib/trendingStats";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

// 比較期間。週次(7日)だと開示の少ない週に振り回され、暦月だと月初は「数日 vs 1か月」の
// 比較になってしまうため、常に同じ長さで比べられる30日固定にする。
const WINDOW_DAYS = 30;
const RANKING_COUNT = 10;

const url = `${SITE_URL}/trending`;
const title = "クジラが急増した銘柄";

export const metadata: Metadata = {
  title,
  description: `直近${WINDOW_DAYS}日間で大量保有報告書の開示が増えた銘柄を、その前の${WINDOW_DAYS}日間と比べた増加件数順にランキング。EDINETの開示データをもとに毎日更新しています。`,
  alternates: { canonical: url },
  openGraph: { title, url },
};

function daysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

export default async function TrendingPage() {
  const currentFrom = daysAgo(WINDOW_DAYS - 1);
  const rangeFrom = daysAgo(WINDOW_DAYS * 2 - 1);
  const rangeTo = daysAgo(0);

  const [rows, stocksWithArticles, monthlyCounts] = await Promise.all([
    getHoldingsInRange(rangeFrom, rangeTo),
    getAllStocksForIndex().catch(() => []),
    getMonthlyDisclosureCounts().catch(() => []),
  ]);

  const trendingIssuers = buildTrendingIssuers(rows, currentFrom, RANKING_COUNT);

  // 銘柄ページ(/stocks/[code])は記事がある銘柄にしか存在しないため、
  // 記事の無い銘柄はリンクにせずテキストのまま出す（404へのリンクを作らない）。
  const codesWithArticles = new Set(stocksWithArticles.map((s) => s.stockCode));

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  // 可視コンテンツと合わせ、実際にリンクしている銘柄（記事がある銘柄）のみItemList化する。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `大量保有報告書の開示が増えた銘柄（直近${WINDOW_DAYS}日）`,
    itemListElement: trendingIssuers
      .filter((entry) => codesWithArticles.has(entry.key))
      .map((entry, index) => ({
        "@type": "ListItem",
        position: index + 1,
        name: entry.label,
        url: `${SITE_URL}/stocks/${entry.key}`,
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
        <span className="text-foreground/70">{title}</span>
      </nav>

      <div className="mb-8">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}が集計した、直近{WINDOW_DAYS}日間（{formatDate(currentFrom)}〜
          {formatDate(rangeTo)}）にEDINETへ提出された大量保有報告書・変更報告書の件数を、
          その前の{WINDOW_DAYS}日間と比べたランキングです。増加件数の多い順に並べています。
          複数の大口投資家が短期間に集まっている銘柄ほど上位に来ます。
          「NEW」は前の{WINDOW_DAYS}日間には開示が無く、この期間に初めて動きが出た銘柄です。
          件数は開示の回数であり、金額の大きさとは別の指標である点にご注意ください
          （金額規模は各銘柄のページで個別の記事としてご覧いただけます）。
        </p>
      </div>

      <section className="mb-10">
        {trendingIssuers.length === 0 ? (
          <p className="text-sm text-foreground/60">
            直近{WINDOW_DAYS}日間で前期間より開示が増えた銘柄はありません。
          </p>
        ) : (
          <TrendingTable
            entries={trendingIssuers}
            headLabel="銘柄"
            windowDays={WINDOW_DAYS}
            hrefOf={(entry) => (codesWithArticles.has(entry.key) ? `/stocks/${entry.key}` : null)}
          />
        )}
      </section>

      {monthlyCounts.length >= 2 && (
        <section className="mb-10">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">月別の開示件数トレンド</h2>
          <p className="mb-2 text-sm text-foreground/60">
            市場全体で大口投資家の動き（開示）が増えているのか減っているのかを月単位で見られます。
          </p>
          <DisclosureTrendChart
            bars={monthlyCounts.map((c) => ({
              key: c.month,
              axisLabel: c.month.slice(2).replace("-", "/"),
              tableLabel: formatMonth(c.month),
              count: c.count,
              isPartial: c.isPartial,
            }))}
            ariaLabel="月別のEDINET大量保有報告書 開示件数の棒グラフ"
            caption="EDINETに提出された大量保有・変更・訂正報告書の月別件数。"
            periodHeadLabel="月"
            partialNote="薄い棒は当月（集計中）です。"
          />
        </section>
      )}

      <nav className="flex flex-wrap gap-x-6 gap-y-2 border-t border-rule pt-6 text-sm">
        <Link href="/ranking/trending" className="text-brand-blue hover:underline">
          開示が急増した投資家を見る ›
        </Link>
        <Link href="/disclosures" className="text-brand-blue hover:underline">
          開示速報で個別の開示を見る ›
        </Link>
        <Link href="/weekly" className="text-brand-blue hover:underline">
          週次トレンドを見る ›
        </Link>
        <Link href="/monthly" className="text-brand-blue hover:underline">
          月別アーカイブで過去の動きを見る ›
        </Link>
      </nav>
      <AdUnit placement="bottom" />
    </div>
  );
}
