import type { Metadata } from "next";
import Link from "next/link";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { formatDate } from "@/lib/format";
import { getAllStocksForIndex } from "@/lib/microcms";
import { getHoldingsInRange } from "@/lib/investors";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { buildTrendingFilers, buildTrendingIssuers, type TrendingEntry } from "@/lib/trendingStats";

export const revalidate = 300;

// 比較期間。週次(7日)だと開示の少ない週に振り回され、暦月だと月初は「数日 vs 1か月」の
// 比較になってしまうため、常に同じ長さで比べられる30日固定にする。
const WINDOW_DAYS = 30;
const RANKING_COUNT = 10;

const url = `${SITE_URL}/trending`;
const title = "クジラが急増した銘柄・投資家";

export const metadata: Metadata = {
  title,
  description: `直近${WINDOW_DAYS}日間で大量保有報告書の開示が増えた銘柄・投資家を、その前の${WINDOW_DAYS}日間と比べた増加件数順にランキング。EDINETの開示データをもとに毎日更新しています。`,
  alternates: { canonical: url },
  openGraph: { title, url },
};

function daysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

function TrendingTable({
  entries,
  headLabel,
  hrefOf,
}: {
  entries: TrendingEntry[];
  headLabel: string;
  hrefOf: (entry: TrendingEntry) => string | null;
}) {
  return (
    <TableContainer>
      <Table size="small" sx={{ minWidth: 420, "& .MuiTableCell-root": { borderColor: "divider" } }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ color: "text.secondary" }}>{headLabel}</TableCell>
            <TableCell align="right" sx={{ color: "text.secondary" }}>直近{WINDOW_DAYS}日</TableCell>
            <TableCell align="right" sx={{ color: "text.secondary" }}>前{WINDOW_DAYS}日</TableCell>
            <TableCell align="right" sx={{ color: "text.secondary" }}>増加</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {entries.map((entry) => {
            const href = hrefOf(entry);
            return (
              <TableRow key={entry.key}>
                <TableCell>
                  {href ? (
                    <Link href={href} className="text-brand-blue hover:underline">
                      {entry.label}
                    </Link>
                  ) : (
                    <span className="text-foreground/80">{entry.label}</span>
                  )}
                  {entry.isNew && (
                    <span className="kicker ml-2 whitespace-nowrap text-brand-gold">NEW</span>
                  )}
                </TableCell>
                <TableCell align="right" sx={{ fontWeight: 700, color: "primary.main" }}>
                  {entry.count}件
                </TableCell>
                <TableCell align="right" sx={{ color: "text.disabled" }}>
                  {entry.prevCount}件
                </TableCell>
                <TableCell align="right" sx={{ color: "text.secondary" }}>+{entry.delta}件</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default async function TrendingPage() {
  const currentFrom = daysAgo(WINDOW_DAYS - 1);
  const rangeFrom = daysAgo(WINDOW_DAYS * 2 - 1);
  const rangeTo = daysAgo(0);

  const [rows, stocksWithArticles] = await Promise.all([
    getHoldingsInRange(rangeFrom, rangeTo),
    getAllStocksForIndex(),
  ]);

  const trendingIssuers = buildTrendingIssuers(rows, currentFrom, RANKING_COUNT);
  const trendingFilers = buildTrendingFilers(rows, currentFrom, RANKING_COUNT);

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
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">
          クジラが急増した銘柄・投資家
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}が集計した、直近{WINDOW_DAYS}日間（{formatDate(currentFrom)}〜
          {formatDate(rangeTo)}）にEDINETへ提出された大量保有報告書・変更報告書の件数を、
          その前の{WINDOW_DAYS}日間と比べたランキングです。増加件数の多い順に並べています。
          「NEW」は前の{WINDOW_DAYS}日間には開示が無く、この期間に初めて動きが出た対象です。
          件数は開示の回数であり、金額の大きさとは別の指標である点にご注意ください
          （金額規模は各銘柄・投資家のページで個別の記事としてご覧いただけます）。
        </p>
      </div>

      <section className="mb-10">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">開示が増えた銘柄</h2>
        <p className="mb-4 text-sm text-foreground/60">
          複数の大口投資家が短期間に集まっている銘柄ほど上位に来ます。
        </p>
        {trendingIssuers.length === 0 ? (
          <p className="text-sm text-foreground/60">
            直近{WINDOW_DAYS}日間で前期間より開示が増えた銘柄はありません。
          </p>
        ) : (
          <TrendingTable
            entries={trendingIssuers}
            headLabel="銘柄"
            hrefOf={(entry) => (codesWithArticles.has(entry.key) ? `/stocks/${entry.key}` : null)}
          />
        )}
      </section>

      <section className="mb-10">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">開示が増えた投資家</h2>
        <p className="mb-4 text-sm text-foreground/60">
          この期間に動きを活発化させた投資家です。名前をクリックすると保有銘柄と保有比率の推移を確認できます。
        </p>
        {trendingFilers.length === 0 ? (
          <p className="text-sm text-foreground/60">
            直近{WINDOW_DAYS}日間で前期間より開示が増えた投資家はありません。
          </p>
        ) : (
          <TrendingTable
            entries={trendingFilers}
            headLabel="投資家"
            hrefOf={(entry) => `/investors/${encodeURIComponent(entry.key)}`}
          />
        )}
      </section>

      <nav className="flex flex-wrap gap-x-6 gap-y-2 border-t border-rule pt-6 text-sm">
        <Link href="/weekly" className="text-brand-blue hover:underline">
          今週の動きを見る ›
        </Link>
        <Link href="/monthly" className="text-brand-blue hover:underline">
          月別アーカイブで過去の動きを見る ›
        </Link>
      </nav>
    </div>
  );
}
