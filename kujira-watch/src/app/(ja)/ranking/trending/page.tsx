import type { Metadata } from "next";
import Link from "next/link";
import { formatDate } from "@/lib/format";
import { getHoldingsInRange } from "@/lib/investors";
import { SITE_URL } from "@/lib/site";
import { buildTrendingFilers } from "@/lib/trendingStats";
import RankingTabNav from "@/components/RankingTabNav";
import TrendingTable from "@/components/TrendingTable";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

// /trendingの銘柄ランキングと同じ期間比較（30日 vs その前30日）。
const WINDOW_DAYS = 30;
const RANKING_COUNT = 10;

const url = `${SITE_URL}/ranking/trending`;
const title = "開示急増投資家ランキング";
const description =
  `直近${WINDOW_DAYS}日間でEDINET大量保有報告書の開示が増えた投資家を、その前の${WINDOW_DAYS}日間と` +
  "比べた増加件数順にランキング。いま動きを活発化させている大口投資家が分かります。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: url },
  openGraph: { title, description, url },
};

function daysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

export default async function TrendingInvestorsRankingPage() {
  const currentFrom = daysAgo(WINDOW_DAYS - 1);
  const rangeFrom = daysAgo(WINDOW_DAYS * 2 - 1);
  const rangeTo = daysAgo(0);

  const rows = await getHoldingsInRange(rangeFrom, rangeTo);
  const trendingFilers = buildTrendingFilers(rows, currentFrom, RANKING_COUNT);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "投資家ランキング", item: `${SITE_URL}/ranking` },
      { "@type": "ListItem", position: 3, name: title, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: title,
    itemListElement: trendingFilers.map((entry, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: entry.label,
      url: `${SITE_URL}/investors/${encodeURIComponent(entry.key)}`,
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
        <Link href="/ranking" className="hover:text-brand-blue">投資家ランキング</Link>
        {" / "}
        <span className="text-foreground/70">{title}</span>
      </nav>
      {/* タブ切替時に見出しの高さ・位置が変わらないよう、h1は全ランキング共通（/rankingと同じ構成）。 */}
      <h1 className="mb-3 text-2xl font-bold text-brand-navy sm:text-3xl">投資家ランキング</h1>
      <RankingTabNav current="/ranking/trending" />
      <h2 className="mb-1 text-lg font-bold text-brand-navy">{title}</h2>
      <p className="mb-2 text-sm text-foreground/50">
        直近{WINDOW_DAYS}日間（{formatDate(currentFrom)}〜{formatDate(rangeTo)}）にEDINETへ提出された
        大量保有報告書・変更報告書の件数を投資家別に集計し、その前の{WINDOW_DAYS}日間と比べて
        増加件数の多い順に並べたランキングです。この期間に動きを活発化させた投資家が分かります。
        「NEW」は前の{WINDOW_DAYS}日間には開示が無く、この期間に初めて動きが出た投資家です。
      </p>
      <p className="mb-6 text-xs text-foreground/40">
        ※件数は開示の回数であり、金額の大きさとは別の指標です。投資家名をクリックすると
        保有銘柄と保有比率の推移を確認できます。投資助言ではありません。
      </p>
      {trendingFilers.length === 0 ? (
        <p className="text-sm text-foreground/60">
          直近{WINDOW_DAYS}日間で前期間より開示が増えた投資家はありません。
        </p>
      ) : (
        <TrendingTable
          entries={trendingFilers}
          headLabel="投資家"
          windowDays={WINDOW_DAYS}
          hrefOf={(entry) => `/investors/${encodeURIComponent(entry.key)}`}
        />
      )}
      <nav className="mt-8 flex flex-wrap gap-x-6 gap-y-2 border-t border-rule pt-6 text-sm">
        <Link href="/trending" className="text-brand-blue hover:underline">
          開示が急増した銘柄を見る ›
        </Link>
        <Link href="/disclosures" className="text-brand-blue hover:underline">
          開示速報で個別の開示を見る ›
        </Link>
      </nav>
      <AdUnit placement="bottom" />
    </div>
  );
}
