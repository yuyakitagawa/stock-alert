import type { Metadata } from "next";
import Link from "next/link";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import DailyDigestList from "@/components/DailyDigestList";
import DisclosureTrendChart from "@/components/DisclosureTrendChart";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import {
  getPreviousPeriodArticles,
  getRecentArticleDigests,
  getRecentArticles,
  type ArticleDigest,
} from "@/lib/microcms";
import { getWeeklyDisclosureCounts } from "@/lib/disclosures";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { formatAmountParts, formatDate, formatDealAmount, isSellArticle } from "@/lib/format";
import { buildWeeklySummary } from "@/lib/weeklyStats";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import AdUnit from "@/components/AdUnit";

// 「大口投資家の動きを教えて」等の包括的な検索・LLMクエリに直答するための集約ページ。
// 直近7日間のサマリーに加え、週ごとの推移（開示件数・売買金額）を出して
// 「今が多いのか少ないのか」のトレンドが分かるようにする（2026-08-18に週次トレンド化）。
const WINDOW_DAYS = 7;

// 週次トレンドの表示期間。開示件数はEDINETデータ（1年分ある）から13週、
// 売買金額は記事データ（2026-07開始）から最大8週。
const TREND_WEEKS = 13;
const AMOUNT_TREND_WEEKS = 8;

// 一覧を全件カード表示すると縦に長くなりすぎるため、金額規模が大きい上位3件のみ
// 「注目」として本文付きで見せ、残りは取引日ごとに/date/[date]へのリンクへ集約する。
const FEATURED_COUNT = 3;

function formatSigned(value: number, unit: string): string {
  return `${value >= 0 ? "+" : ""}${value.toLocaleString("ja-JP")}${unit}`;
}

// 月曜始まりの暦週の開始日（YYYY-MM-DD）。dealDateはISO日時のこともあるため日付部分だけ使う。
function weekStartOf(dateStr: string): string {
  const d = new Date(`${dateStr.slice(0, 10)}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}

// "8/11〜8/17" 形式の週ラベル。
function weekRangeLabel(weekStart: string): string {
  const start = new Date(`${weekStart}T00:00:00Z`);
  const end = new Date(start);
  end.setUTCDate(end.getUTCDate() + 6);
  const fmt = (d: Date) => `${d.getUTCMonth() + 1}/${d.getUTCDate()}`;
  return `${fmt(start)}〜${fmt(end)}`;
}

type WeeklyAmountRow = {
  weekStart: string;
  count: number;
  buyAmount: number;
  sellAmount: number;
  isPartial: boolean;
};

// 記事ダイジェストを月曜始まりの暦週で集計する。記事データが無い最古側の空週は出さない。
function buildWeeklyAmountRows(digests: ArticleDigest[], weeks: number): WeeklyAmountRow[] {
  const currentWeekStart = weekStartOf(new Date().toISOString().slice(0, 10));
  const byWeek = new Map<string, WeeklyAmountRow>();
  for (const digest of digests) {
    const weekStart = weekStartOf(digest.dealDate);
    const row = byWeek.get(weekStart) ?? {
      weekStart,
      count: 0,
      buyAmount: 0,
      sellAmount: 0,
      isPartial: weekStart === currentWeekStart,
    };
    row.count += 1;
    if (isSellArticle(digest.tags)) {
      row.sellAmount += digest.dealAmount;
    } else {
      row.buyAmount += digest.dealAmount;
    }
    byWeek.set(weekStart, row);
  }
  return [...byWeek.values()]
    .sort((a, b) => (a.weekStart < b.weekStart ? 1 : -1))
    .slice(0, weeks);
}

export async function generateMetadata(): Promise<Metadata> {
  const { contents } = await getRecentArticles(WINDOW_DAYS);
  const title = "大口投資家の週次トレンド";
  const description =
    `EDINET大量保有報告書をもとにした大口投資家（クジラ）の週次トレンド。直近${WINDOW_DAYS}日間の` +
    `${contents.length}件の開示まとめに加え、週ごとの開示件数・売買金額の推移を確認できます。`;
  const url = `${SITE_URL}/weekly`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function WeeklyDigestPage() {
  const [{ contents }, { contents: previousContents }, weeklyCounts, digests] = await Promise.all([
    getRecentArticles(WINDOW_DAYS),
    getPreviousPeriodArticles(WINDOW_DAYS),
    getWeeklyDisclosureCounts(TREND_WEEKS).catch(() => []),
    getRecentArticleDigests(AMOUNT_TREND_WEEKS * 7).catch(() => []),
  ]);
  const url = `${SITE_URL}/weekly`;

  const oldestDate = contents.length > 0 ? contents[contents.length - 1].dealDate : null;
  const newestDate = contents.length > 0 ? contents[0].dealDate : null;
  const summary = buildWeeklySummary(contents);
  const previousSummary = buildWeeklySummary(previousContents);
  const countDelta = summary.totalCount - previousSummary.totalCount;
  const amountDelta = summary.totalAmount - previousSummary.totalAmount;
  const amountDeltaPct = previousSummary.totalAmount > 0 ? (amountDelta / previousSummary.totalAmount) * 100 : null;
  const netAmount = summary.buyAmount - summary.sellAmount;
  const netLabel = netAmount >= 0 ? "買い越し" : "売り越し";
  const featured = [...contents].sort((a, b) => b.dealAmount - a.dealAmount).slice(0, FEATURED_COUNT);
  const dateGroups = groupArticlesByDealDate(contents);
  const dateGroupMap = new Map(dateGroups.map((g) => [g.date, g]));
  const amountRows = buildWeeklyAmountRows(digests, AMOUNT_TREND_WEEKS);
  // 開示が無い日も含めて直近7日分を並べる（dateGroupsは開示があった日しか持たないため）。
  const last7Dates = Array.from({ length: WINDOW_DAYS }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - i);
    return d.toISOString().slice(0, 10);
  });

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "大口投資家の週次トレンド", item: url },
    ],
  };

  // 可視コンテンツと構造化データを一致させるため、本文で実際にリンクしている要素
  // （注目記事の個別ページ＋取引日別アーカイブページ）だけをItemList化する。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "大口投資家の週次トレンド",
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
        <span className="text-foreground/70">大口投資家の週次トレンド</span>
      </nav>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">大口投資家の週次トレンド</h1>
        {contents.length > 0 && oldestDate && newestDate ? (
          <>
            <p className="mt-3 text-sm leading-relaxed text-foreground/70">
              {SITE_NAME}がEDINET大量保有報告書をもとに集計した、大口投資家（クジラ）の
              週ごとの動きです。直近{WINDOW_DAYS}日間（{formatDate(oldestDate)}〜
              {formatDate(newestDate)}）のサマリーと、週別の開示件数・売買金額の推移を掲載しています。
            </p>
            <Box sx={{ mt: 2, display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 2 }}>
              <Card variant="outlined" sx={{ borderLeft: 4, borderLeftColor: "brand.gold", bgcolor: "action.hover", p: 2, borderColor: "divider" }}>
                <Typography variant="overline" sx={{ color: "text.secondary" }}>開示件数（直近7日）</Typography>
                <Typography
                  variant="h4"
                  sx={{ mt: 0.5, fontWeight: 700, color: "primary.main", fontSize: { xs: "1.75rem", sm: "2.125rem" } }}
                >
                  {summary.totalCount}
                  <Typography component="span" variant="body1" sx={{ ml: 0.5 }}>件</Typography>
                </Typography>
                <Typography variant="caption" sx={{ mt: 0.5, display: "block", color: "text.secondary" }}>
                  前週比 {formatSigned(countDelta, "件")}（前7日間{previousSummary.totalCount}件）
                </Typography>
              </Card>
              <Card variant="outlined" sx={{ borderLeft: 4, borderLeftColor: "brand.gold", bgcolor: "action.hover", p: 2, borderColor: "divider" }}>
                <Typography variant="overline" sx={{ color: "text.secondary" }}>推定取引金額（直近7日）</Typography>
                {/* 「4,083億円」をh4一体で出すと375pxのタイル幅で折り返して高さが揃わない。
                    開示件数タイルと同じ「数字+単位」構造に統一し、1兆円以上は兆円へ繰り上げる。 */}
                <Typography
                  variant="h4"
                  sx={{
                    mt: 0.5,
                    fontWeight: 700,
                    color: "primary.main",
                    // 375pxのタイル幅でも「数字+単位」が1行に収まるサイズ（左の件数タイルと揃える）。
                    fontSize: { xs: "1.75rem", sm: "2.125rem" },
                    whiteSpace: "nowrap",
                  }}
                >
                  {formatAmountParts(summary.totalAmount).value}
                  <Typography component="span" variant="body1" sx={{ ml: 0.5 }}>
                    {formatAmountParts(summary.totalAmount).unit}
                  </Typography>
                </Typography>
                <Typography variant="caption" sx={{ mt: 0.5, display: "block", color: "text.secondary" }}>
                  前週比{" "}
                  {amountDeltaPct !== null
                    ? formatSigned(Number(amountDeltaPct.toFixed(1)), "%")
                    : formatSigned(amountDelta, "億円")}
                  （前7日間{formatDealAmount(previousSummary.totalAmount)}）
                </Typography>
              </Card>
            </Box>
            <p className="mt-2 text-xs text-foreground/50">
              ※推定取引金額は取得・売却双方向を合算した規模で、資金の純流入額ではありません。
            </p>
          </>
        ) : (
          <p className="mt-3 text-sm leading-relaxed text-foreground/70">
            直近{WINDOW_DAYS}日間はEDINET大量保有報告書の新規開示がありませんでした。
          </p>
        )}
      </div>

      {weeklyCounts.length >= 2 && (
        <section className="mb-10">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">週別の開示件数トレンド</h2>
          <p className="mb-2 text-sm text-foreground/60">
            大口投資家の動き（EDINET開示）が週単位で増えているのか減っているのかを確認できます。
          </p>
          <DisclosureTrendChart
            bars={weeklyCounts.map((w) => ({
              key: w.weekStart,
              axisLabel: weekRangeLabel(w.weekStart).split("〜")[0],
              tableLabel: weekRangeLabel(w.weekStart),
              count: w.count,
              isPartial: w.isPartial,
            }))}
            ariaLabel="週別のEDINET大量保有報告書 開示件数の棒グラフ"
            caption="EDINETに提出された大量保有・変更・訂正報告書の週別件数（月曜始まり）。"
            periodHeadLabel="週"
            partialNote="薄い棒は今週（集計中）です。"
          />
        </section>
      )}

      {amountRows.length >= 2 && (
        <section className="mb-10">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">週別の売買金額トレンド</h2>
          <p className="mb-2 text-sm text-foreground/60">
            解説記事化した開示の推定金額を週ごとに買い・売りへ分けた推移です。
            差し引きがプラスの週は買い越し、マイナスの週は売り越しです。
          </p>
          <TableContainer>
            <Table size="small" sx={{ minWidth: 420, "& .MuiTableCell-root": { borderColor: "divider" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: "text.secondary" }}>週</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>件数</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>買い</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>売り</TableCell>
                  <TableCell align="right" sx={{ color: "text.secondary" }}>差し引き</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {amountRows.map((row) => {
                  const net = row.buyAmount - row.sellAmount;
                  return (
                    <TableRow key={row.weekStart}>
                      <TableCell sx={{ color: "primary.main", whiteSpace: "nowrap" }}>
                        {weekRangeLabel(row.weekStart)}
                        {row.isPartial && (
                          <Box component="span" sx={{ ml: 0.5, fontSize: "0.6875rem", color: "text.disabled" }}>
                            集計中
                          </Box>
                        )}
                      </TableCell>
                      <TableCell align="right" sx={{ color: "text.secondary" }}>{row.count}件</TableCell>
                      <TableCell align="right" sx={{ color: "text.secondary" }}>
                        {formatDealAmount(row.buyAmount)}
                      </TableCell>
                      <TableCell align="right" sx={{ color: "text.secondary" }}>
                        {formatDealAmount(row.sellAmount)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ fontWeight: 700, color: net >= 0 ? "success.main" : "error.main", whiteSpace: "nowrap" }}
                      >
                        {net >= 0 ? "+" : "-"}
                        {formatDealAmount(Math.abs(net))}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
          <p className="mt-2 text-xs text-foreground/50">
            ※推定金額は発行済株式数×株価×保有比率の変化幅から概算した参考値です。
          </p>
        </section>
      )}

      {contents.length > 0 && (
        <section className="mb-10 border-y border-rule py-6">
          <h2 className="text-lg font-bold text-brand-navy">直近7日間のポイント</h2>
          <Box sx={{ mt: 2, display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 2 }}>
            <Card variant="outlined" sx={{ bgcolor: "action.hover", p: 2, borderColor: "divider" }}>
              <Typography variant="overline" sx={{ color: "text.secondary" }}>買い</Typography>
              <Typography variant="h5" sx={{ mt: 0.5, fontWeight: 700, color: "primary.main" }}>
                {formatDealAmount(summary.buyAmount)}
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.25, color: "text.secondary" }}>{summary.buyCount}件</Typography>
            </Card>
            <Card variant="outlined" sx={{ bgcolor: "action.hover", p: 2, borderColor: "divider" }}>
              <Typography variant="overline" sx={{ color: "text.secondary" }}>売り</Typography>
              <Typography variant="h5" sx={{ mt: 0.5, fontWeight: 700, color: "primary.main" }}>
                {formatDealAmount(summary.sellAmount)}
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.25, color: "text.secondary" }}>{summary.sellCount}件</Typography>
            </Card>
          </Box>
          <p className="mt-4 text-sm leading-relaxed text-foreground/70">
            金額ベースでは買いが{formatDealAmount(summary.buyAmount)}、売りが
            {formatDealAmount(summary.sellAmount)}で、差し引き{formatDealAmount(Math.abs(netAmount))}の
            {netLabel}でした（複数の開示を合算した推定値のため、実際の資金フローとは異なります）。
          </p>

          {summary.categoryBreakdown.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="overline" sx={{ display: "block", mb: 1, color: "text.secondary" }}>
                投資家分類別の内訳
              </Typography>
              <TableContainer>
                <Table size="small" sx={{ minWidth: 420, "& .MuiTableCell-root": { borderColor: "divider" } }}>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: "text.secondary" }}>投資家分類</TableCell>
                      <TableCell align="right" sx={{ color: "text.secondary" }}>件数</TableCell>
                      <TableCell align="right" sx={{ color: "text.secondary" }}>推定金額</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {summary.categoryBreakdown.map((c) => (
                      <TableRow key={c.dealType}>
                        <TableCell title={DEAL_TYPE_DESCRIPTIONS[c.dealType]} sx={{ color: "primary.main" }}>
                          {c.dealType}
                        </TableCell>
                        <TableCell align="right" sx={{ color: "text.secondary" }}>{c.count}件</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 700, color: "primary.main" }}>
                          {formatDealAmount(c.amount)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </section>
      )}

      {featured.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">注目の取引</h2>
          {summary.topStocks.length > 0 && (
            <p className="mb-4 text-sm leading-relaxed text-foreground/70">
              個別銘柄では
              {summary.topStocks.map((s, i) => (
                <span key={s.stockCode}>
                  {i > 0 && "、"}
                  <Link href={`/stocks/${s.stockCode}`} className="font-bold text-brand-blue hover:underline">
                    {s.stockName}（{s.stockCode}）
                  </Link>
                  （{s.count}件・{formatDealAmount(s.amount)}）
                </span>
              ))}
              への開示が目立ちました。
            </p>
          )}
          <div className="space-y-4">
            {featured.map((article, i) => (
              <FeaturedArticleCard key={article.id} article={article} rank={i + 1} />
            ))}
          </div>
        </section>
      )}

      {contents.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-bold text-brand-navy">日別の記事一覧</h2>
          <DailyDigestList
            days={last7Dates.map((date) => {
              const group = dateGroupMap.get(date);
              return {
                date,
                label: group?.label ?? formatDate(date),
                count: group?.articles.length ?? 0,
                amount: group ? group.articles.reduce((sum, a) => sum + a.dealAmount, 0) : 0,
              };
            })}
          />
        </section>
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}
