import type { Metadata } from "next";
import Link from "next/link";
import { formatMonth } from "@/lib/format";
import { getHoldingsInRange } from "@/lib/investors";
import { getAllListedCodes, getCompanyBriefs } from "@/lib/companyInfo";
import { getMonthlyDisclosureCounts } from "@/lib/disclosures";
import DisclosureTrendChart from "@/components/DisclosureTrendChart";
import TrendingDirectionTable from "@/components/TrendingDirectionTable";
import { SITE_URL } from "@/lib/site";
import { buildTrendingIssuers, selectDirection } from "@/lib/trendingStats";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

// 比較期間。週次(7日)だと開示の少ない週に振り回され、暦月だと月初は「数日 vs 1か月」の
// 比較になってしまうため、常に同じ長さで比べられる30日固定にする。
const WINDOW_DAYS = 30;

// 構造化データに載せる件数。オートページャーで追加描画される分はクライアント側の
// 処理なので、クロール時点でサーバーが返すHTML（TrendingTableのINITIAL_COUNT）に合わせる。
const JSON_LD_COUNT = 30;

const url = `${SITE_URL}/trending`;
const title = "銘柄ランキング";

export const metadata: Metadata = {
  title,
  description: `直近${WINDOW_DAYS}日間で大量保有報告書の開示が増えた銘柄を、その前の${WINDOW_DAYS}日間と比べた増加件数順にランキング。買い・売り・両方で絞り込めます。EDINETの開示データをもとに毎日更新しています。`,
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

  const [rows, listedCodes, monthlyCounts] = await Promise.all([
    getHoldingsInRange(rangeFrom, rangeTo),
    getAllListedCodes().catch(() => new Set<string>()),
    getMonthlyDisclosureCounts().catch(() => []),
  ]);

  // 件数制限なし。直近30日で開示が増えた銘柄をすべて出す（買い・売り・両方のいずれかで増えたもの）。
  const trendingIssuers = buildTrendingIssuers(rows, currentFrom);

  // 社名と証券コードだけでは何の会社か分からないため、事業内容を1行添える。
  // 事業内容(jpx_stock_list.description)は未生成の銘柄があるので、その場合は業種で代替する。
  const briefs = await getCompanyBriefs(trendingIssuers.map((entry) => entry.key));
  const noteOf = (code: string): string | null => {
    const brief = briefs.get(code);
    return brief?.description ?? brief?.sector ?? null;
  };

  // 銘柄ページ(/stocks/[code])は上場銘柄マスターに載っていれば解説記事が無くても
  // 開示履歴＋会社情報で成立する。マスターに無いコード（上場廃止等）だけ
  // リンクにせずテキストのまま出す（404へのリンクを作らない）。

  // href・noteはサーバー側で解決してから渡す（TrendingDirectionTableはクライアント
  // コンポーネントなので関数propsを境界を越えて渡せない）。
  const trendingItems = trendingIssuers.map((entry) => ({
    ...entry,
    href: listedCodes.has(entry.key) ? `/stocks/${entry.key}` : null,
    note: noteOf(entry.key),
  }));

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  // 可視コンテンツと合わせ、実際にリンクしている銘柄のみItemList化する。
  // 絞り込みの初期値は「買い」なので、SSRされるHTMLと同じく買いの一覧を載せる。
  const defaultViewItems = selectDirection(trendingItems, "buy");
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `買いの大量保有報告書の開示が増えた銘柄（直近${WINDOW_DAYS}日）`,
    itemListElement: defaultViewItems
      .slice(0, JSON_LD_COUNT)
      .filter((entry) => entry.href !== null)
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
          大口投資家の取引（大量保有報告書の提出）が直近{WINDOW_DAYS}日間で前の{WINDOW_DAYS}日間より増えた銘柄のランキングです。
          保有比率が増えた「買い」（初期表示）・減った「売り」・その両方で絞り込めます。
          <br />
          投資家版は
          <Link href="/ranking/trending" className="text-brand-blue hover:underline">
            開示急増投資家ランキング
          </Link>
          へ。詳しい見方は
          <Link href="/faq/usage" className="text-brand-blue hover:underline">
            よくある質問
          </Link>
          をご覧ください。
        </p>
      </div>

      <section className="mb-10">
        <TrendingDirectionTable items={trendingItems} windowDays={WINDOW_DAYS} />
      </section>

      {monthlyCounts.length >= 2 && (
        <section className="mb-10">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">月別の開示件数トレンド</h2>
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

      <AdUnit placement="bottom" />
    </div>
  );
}
