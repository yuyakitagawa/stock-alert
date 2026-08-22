import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import ActionButton from "@/components/ActionButton";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { formatDealAmount, formatMonth } from "@/lib/format";
import { getAllMonthsForIndex, getArticlesByMonth, MONTH_PATTERN } from "@/lib/microcms";
import { getFilerIdMap, getFilerNamesByStockAndDate, investorPath } from "@/lib/investors";
import { SITE_URL } from "@/lib/site";
import { buildFilerRanking, buildStockRanking, buildWeeklySummary } from "@/lib/weeklyStats";
import AdUnit from "@/components/AdUnit";

export const revalidate = 60;

// 「その月の主役」を示すランキングの表示件数。日次(/date)・週次(/weekly)より
// 母数が多いぶん、上位3件では月の全体像が見えないため10件まで見せる。
const RANKING_COUNT = 10;
const FEATURED_COUNT = 3;

type Props = {
  params: Promise<{ month: string }>;
};

// 記事のある月をビルド時に事前生成する。generateStaticParamsが無いと動的ルートとして
// 毎回サーバーレンダリングされ（実測: 本番で x-vercel-cache: MISS、TTFB約0.9〜1.2秒）、
// ページ単位のrevalidateもCDNに載らないため。月が増えた場合は
// dynamicParams（既定true）により初回アクセス時にオンデマンド生成される。
export async function generateStaticParams() {
  const months = await getAllMonthsForIndex();
  return months.map((m) => ({ month: m.month }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { month } = await params;
  if (!MONTH_PATTERN.test(month)) return {};

  const { contents } = await getArticlesByMonth(month);
  if (contents.length === 0) return {};

  const label = formatMonth(month);
  const summary = buildWeeklySummary(contents);
  const topStock = buildStockRanking(contents, 1)[0];
  const title = `${label}の大口投資家の動きまとめ`;
  const description =
    `${label}にEDINETで開示された大量保有・変更報告書${contents.length}件（推定${formatDealAmount(summary.totalAmount)}）を集計。` +
    (topStock ? `最も動きが大きかったのは${topStock.stockName}（${topStock.stockCode}）。` : "") +
    "投資家別・銘柄別のランキングと日別の一覧を掲載しています。";

  return {
    title,
    description,
    alternates: { canonical: `${SITE_URL}/monthly/${month}` },
    openGraph: { title, description, url: `${SITE_URL}/monthly/${month}` },
  };
}

export default async function MonthlyArchivePage({ params }: Props) {
  const { month } = await params;
  if (!MONTH_PATTERN.test(month)) {
    notFound();
  }

  const [{ contents }, months, filerByKey, filerIds] = await Promise.all([
    getArticlesByMonth(month),
    getAllMonthsForIndex(),
    // disc_dateはtext型のYYYY-MM-DDなので、月末日は31日固定の文字列比較で足りる。
    getFilerNamesByStockAndDate(`${month}-01`, `${month}-31`),
    getFilerIdMap(),
  ]);

  if (contents.length === 0) {
    notFound();
  }

  const label = formatMonth(month);
  const url = `${SITE_URL}/monthly/${month}`;
  const summary = buildWeeklySummary(contents);
  const netAmount = summary.buyAmount - summary.sellAmount;
  const netLabel = netAmount >= 0 ? "買い越し" : "売り越し";
  // 記事側に提出者名のフィールドが無いため（getFilerNamesByStockAndDateの注記を参照）、
  // 「銘柄コード×取引日」で一意に定まる開示だけ提出者を補って集計する。
  const articlesWithFiler = contents.map((article) => ({
    ...article,
    filerName: article.filerName ?? filerByKey.get(`${article.stockCode}|${article.dealDate.slice(0, 10)}`),
  }));
  const attributedCount = articlesWithFiler.filter((a) => a.filerName).length;
  const topFilers = buildFilerRanking(articlesWithFiler, RANKING_COUNT);
  const topStocks = buildStockRanking(contents, RANKING_COUNT);
  const featured = [...contents].sort((a, b) => b.dealAmount - a.dealAmount).slice(0, FEATURED_COUNT);
  const dateGroups = groupArticlesByDealDate(contents);

  // 月一覧は新しい順のため、配列上の「次の要素」が前月にあたる。
  const monthIndex = months.findIndex((m) => m.month === month);
  const olderMonth = monthIndex >= 0 ? months[monthIndex + 1] : undefined;
  const newerMonth = monthIndex > 0 ? months[monthIndex - 1] : undefined;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "月別アーカイブ", item: `${SITE_URL}/monthly` },
      { "@type": "ListItem", position: 3, name: label, item: url },
    ],
  };

  // 可視コンテンツと構造化データを一致させるため、本文で実際にリンクしている要素
  // （注目記事＋取引日別アーカイブ）だけをItemList化する。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${label}の大口投資家の動き`,
    itemListElement: [
      ...featured.map((article, index) => ({
        "@type": "ListItem",
        position: index + 1,
        name: article.title,
        url: `${SITE_URL}/articles/${article.id}`,
      })),
      ...dateGroups.map((group, index) => ({
        "@type": "ListItem",
        position: featured.length + index + 1,
        name: `${group.label}の大口投資家の動き`,
        url: `${SITE_URL}/date/${group.date}`,
      })),
    ],
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
        <Link href="/monthly" className="hover:text-brand-blue">月別アーカイブ</Link>
        {" / "}
        <span className="text-foreground/70">{label}</span>
      </nav>

      <div className="mb-8">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">
          {label}の大口投資家の動き
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          {label}にEDINETで開示された大量保有・変更報告書は{summary.totalCount}件、
          推定取引金額は{formatDealAmount(summary.totalAmount)}でした。金額ベースでは買いが
          {formatDealAmount(summary.buyAmount)}（{summary.buyCount}件）、売りが
          {formatDealAmount(summary.sellAmount)}（{summary.sellCount}件）で、差し引き
          {formatDealAmount(Math.abs(netAmount))}の{netLabel}です
          （複数の開示を合算した推定値のため、実際の資金フローとは異なります）。
        </p>
      </div>

      {topFilers.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">この月に動いた投資家</h2>
          <p className="mb-4 text-sm text-foreground/60">
            開示された取引の推定金額が大きい順（{label}分の合算）。同じ銘柄・同じ日に複数の
            投資家が開示していて提出者を一意に特定できない取引は集計に含めていません
            （{summary.totalCount}件中{attributedCount}件を集計）。
          </p>
          <TableContainer>
            <Table size="small" sx={{ minWidth: 420, "& .MuiTableCell-root": { borderColor: "divider" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: "text.secondary" }}>投資家</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>件数</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>推定金額</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {topFilers.map((filer) => (
                  <TableRow key={filer.filerName}>
                    <TableCell>
                      <Link
                        href={investorPath(filerIds[filer.filerName], filer.filerName)}
                        className="text-brand-blue hover:underline"
                      >
                        {filer.filerName}
                      </Link>
                    </TableCell>
                    <TableCell align="right" sx={{ color: "text.secondary" }}>{filer.count}件</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 700, color: "primary.main" }}>
                      {formatDealAmount(filer.amount)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </section>
      )}

      {topStocks.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">この月に狙われた銘柄</h2>
          <p className="mb-4 text-sm text-foreground/60">
            大口投資家の開示が集まった銘柄を推定金額の大きい順に並べています。
          </p>
          <TableContainer>
            <Table size="small" sx={{ minWidth: 420, "& .MuiTableCell-root": { borderColor: "divider" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: "text.secondary" }}>銘柄</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>件数</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>推定金額</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {topStocks.map((stock) => (
                  <TableRow key={stock.stockCode}>
                    <TableCell>
                      <Link href={`/stocks/${stock.stockCode}`} className="text-brand-blue hover:underline">
                        {stock.stockName}（{stock.stockCode}）
                      </Link>
                    </TableCell>
                    <TableCell align="right" sx={{ color: "text.secondary" }}>{stock.count}件</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 700, color: "primary.main" }}>
                      {formatDealAmount(stock.amount)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </section>
      )}

      {featured.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-4 text-xl font-bold text-brand-navy">{label}の注目取引</h2>
          <div className="space-y-4">
            {featured.map((article, i) => (
              <FeaturedArticleCard key={article.id} article={article} rank={i + 1} />
            ))}
          </div>
        </section>
      )}

      <section className="mb-10">
        <h2 className="mb-4 text-xl font-bold text-brand-navy">日別の記事一覧</h2>
        <div className="divide-y divide-rule border-y border-rule">
          {dateGroups.map((group) => {
            const dayAmount = group.articles.reduce((sum, a) => sum + a.dealAmount, 0);
            return (
              <Link
                key={group.date}
                href={`/date/${group.date}`}
                className="group flex items-center justify-between gap-4 py-4 transition-colors hover:bg-section-tint"
              >
                <div>
                  <p className="font-bold text-brand-navy">{group.label}</p>
                  <p className="mt-0.5 text-sm text-foreground/60">
                    {group.articles.length}件・{formatDealAmount(dayAmount)}
                  </p>
                </div>
                <span className="kicker shrink-0 text-brand-blue transition-colors group-hover:text-brand-navy">
                  この日の記事を見る ›
                </span>
              </Link>
            );
          })}
        </div>
      </section>

      <nav aria-label="前後の月" className="flex flex-wrap items-center justify-between gap-2 border-t border-rule pt-6">
        {olderMonth ? (
          <ActionButton href={`/monthly/${olderMonth.month}`}>
            ‹ {formatMonth(olderMonth.month)}
          </ActionButton>
        ) : (
          <span />
        )}
        <ActionButton href="/monthly">月別アーカイブ一覧</ActionButton>
        {newerMonth ? (
          <ActionButton href={`/monthly/${newerMonth.month}`}>
            {formatMonth(newerMonth.month)} ›
          </ActionButton>
        ) : (
          <span />
        )}
      </nav>
      <AdUnit placement="bottom" />
    </div>
  );
}
