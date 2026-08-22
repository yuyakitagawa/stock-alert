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
import AmountTrendChart from "@/components/AmountTrendChart";
import CategoryWeeklyTrend, {
  type CategoryWeekColumn,
  type CategoryWeeklyRow,
} from "@/components/CategoryWeeklyTrend";
import {
  getPreviousPeriodArticles,
  getRecentArticleDigests,
  getRecentArticles,
  type ArticleDigest,
} from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { formatAmountParts, formatDate, formatDealAmount, isSellArticle } from "@/lib/format";
import type { DealType } from "@/types/article";
import { buildWeeklySummary } from "@/lib/weeklyStats";
import AdUnit from "@/components/AdUnit";

// 「大口投資家の動きを教えて」等の包括的な検索・LLMクエリに直答するための集約ページ。
// 直近7日間のサマリーに加え、週ごとの推移（売買金額・投資家分類別）を出して
// 「今が多いのか少ないのか」のトレンドが分かるようにする（2026-08-18に週次トレンド化）。
const WINDOW_DAYS = 7;

// 週次トレンドの表示期間。記事データ（2026-07開始）から最大8週。
const AMOUNT_TREND_WEEKS = 8;

// 投資家分類別の週次トレンドは行が13分類あり、列（週）を増やすほど横スクロールが伸びる。
// 「今どの分類が活発か」を読むのに必要な直近ぶんだけを、売買金額トレンドと同じ週区切りで出す。
const CATEGORY_TREND_WEEKS = 6;

function formatSigned(value: number, unit: string): string {
  return `${value >= 0 ? "+" : ""}${value.toLocaleString("ja-JP")}${unit}`;
}

// 月曜始まりの暦週の開始日（YYYY-MM-DD）。dealDateはISO日時のこともあるため日付部分だけ使う。
function weekStartOf(dateStr: string): string {
  const d = new Date(`${dateStr.slice(0, 10)}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}

// 「今日」は日本時間で判定する。UTCのまま new Date() を使うと月曜の0〜9時（JST）に
// UTCではまだ日曜で、今週が前週として扱われ最新週がグラフから消える。
function todayJst(): string {
  return new Date().toLocaleDateString("sv-SE", { timeZone: "Asia/Tokyo" });
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

// 記事ダイジェストを月曜始まりの暦週で集計する。開示が1件も無い週も枠として残し
// （グラフの横軸が飛ぶと「その週は少なかった」と読めてしまうため）、記事データが
// まだ無い最古側の空週だけ落とす。新しい週が先頭。
function buildWeeklyAmountRows(digests: ArticleDigest[], weeks: number): WeeklyAmountRow[] {
  const currentWeekStart = weekStartOf(todayJst());
  const byWeek = new Map<string, WeeklyAmountRow>();
  // 直近weeks週ぶんの枠を先に作る（間の週が抜けないように）。
  for (let i = 0; i < weeks; i++) {
    const d = new Date(`${currentWeekStart}T00:00:00Z`);
    d.setUTCDate(d.getUTCDate() - i * 7);
    const weekStart = d.toISOString().slice(0, 10);
    byWeek.set(weekStart, {
      weekStart,
      count: 0,
      buyAmount: 0,
      sellAmount: 0,
      isPartial: weekStart === currentWeekStart,
    });
  }
  for (const digest of digests) {
    const row = byWeek.get(weekStartOf(digest.dealDate));
    if (!row) continue;
    row.count += 1;
    if (isSellArticle(digest.tags)) {
      row.sellAmount += digest.dealAmount;
    } else {
      row.buyAmount += digest.dealAmount;
    }
  }
  const rows = [...byWeek.values()].sort((a, b) => (a.weekStart < b.weekStart ? 1 : -1));
  while (rows.length > 0 && rows[rows.length - 1].count === 0) rows.pop();
  return rows;
}

// 投資家分類×週の集計。週の枠（weekStarts）は売買金額トレンドと同じものを渡し、
// グラフと表で「同じ週」を見ていることが保証されるようにする。古い週→新しい週の並び。
// 買いと売りは打ち消し合ってしまうため合算せず、方向ごとに1回ずつ呼んで別々の表にする。
function buildCategoryWeeklyRows(
  digests: ArticleDigest[],
  weekStarts: string[],
  tone: "buy" | "sell"
): CategoryWeeklyRow[] {
  const weekIndex = new Map(weekStarts.map((weekStart, i) => [weekStart, i]));
  const byCategory = new Map<DealType, CategoryWeeklyRow>();
  for (const digest of digests) {
    const i = weekIndex.get(weekStartOf(digest.dealDate));
    if (i === undefined || !digest.dealType) continue;
    if (isSellArticle(digest.tags) !== (tone === "sell")) continue;
    const row =
      byCategory.get(digest.dealType) ??
      {
        dealType: digest.dealType,
        cells: weekStarts.map(() => ({ count: 0, amount: 0 })),
        total: 0,
      };
    row.cells[i].count += 1;
    row.cells[i].amount += digest.dealAmount;
    row.total += digest.dealAmount;
    byCategory.set(digest.dealType, row);
  }
  return [...byCategory.values()].sort((a, b) => b.total - a.total);
}

export async function generateMetadata(): Promise<Metadata> {
  const { contents } = await getRecentArticles(WINDOW_DAYS);
  const title = "大口投資家の週次トレンド";
  const description =
    `EDINET大量保有報告書をもとにした大口投資家の週次トレンド。直近${WINDOW_DAYS}日間の` +
    `${contents.length}件の開示まとめに加え、週ごとの売買金額・投資家分類別（買い/売り）の推移を確認できます。`;
  const url = `${SITE_URL}/weekly`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function WeeklyDigestPage() {
  const [{ contents }, { contents: previousContents }, digests] = await Promise.all([
    getRecentArticles(WINDOW_DAYS),
    getPreviousPeriodArticles(WINDOW_DAYS),
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
  const amountRows = buildWeeklyAmountRows(digests, AMOUNT_TREND_WEEKS);
  // 分類別トレンドは金額トレンドの週枠の新しい方から必要数だけ切り出す（古い週→新しい週）。
  // グラフと違い表では空の列に意味が無いため、記事データが無い最古側の週は列ごと落とす。
  const categoryWeekRows = amountRows.slice(0, CATEGORY_TREND_WEEKS).reverse();
  while (categoryWeekRows.length > 0 && categoryWeekRows[0].count === 0) categoryWeekRows.shift();
  const categoryWeeks: CategoryWeekColumn[] = categoryWeekRows
    .map((row) => ({
      weekStart: row.weekStart,
      axisLabel: weekRangeLabel(row.weekStart).split("〜")[0],
      tableLabel: weekRangeLabel(row.weekStart),
      isPartial: row.isPartial,
    }));
  const categoryWeekStarts = categoryWeeks.map((w) => w.weekStart);
  const buyCategoryRows = buildCategoryWeeklyRows(digests, categoryWeekStarts, "buy");
  const sellCategoryRows = buildCategoryWeeklyRows(digests, categoryWeekStarts, "sell");
  // 「今どこが活発か」を文章でも言い切る（表を読まない読者・LLM向け）。買い・売りそれぞれの最新週の上位。
  const latestWeek = categoryWeeks[categoryWeeks.length - 1];
  const latestLeaders = (rows: CategoryWeeklyRow[]) =>
    rows
      .map((row) => ({ dealType: row.dealType, ...row.cells[row.cells.length - 1] }))
      .filter((c) => c.count > 0)
      .sort((a, b) => b.amount - a.amount)
      .slice(0, 3);
  const latestBuyLeaders = latestLeaders(buyCategoryRows);
  const latestSellLeaders = latestLeaders(sellCategoryRows);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "大口投資家の週次トレンド", item: url },
    ],
  };

  return (
    <div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
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
              {SITE_NAME}がEDINET大量保有報告書をもとに集計した、大口投資家の
              週ごとの動きです。直近{WINDOW_DAYS}日間（{formatDate(oldestDate)}〜
              {formatDate(newestDate)}）のサマリーと、週別の売買金額・投資家分類別の推移を掲載しています。
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

      {amountRows.length >= 2 && (
        <section className="mb-10">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">週別の売買金額トレンド</h2>
          <p className="mb-2 text-sm text-foreground/60">
            解説記事化した開示の推定金額を週ごとに買い・売りへ分けた推移です。
            ベースラインより上が買い、下が売りで、上に大きく振れた週ほど買い越しです。
          </p>
          {/* グラフは古い週が左＝時系列順。表は他の一覧と揃えて新しい週が上のまま。 */}
          <AmountTrendChart
            bars={[...amountRows].reverse().map((row) => ({
              key: row.weekStart,
              axisLabel: weekRangeLabel(row.weekStart).split("〜")[0],
              tableLabel: weekRangeLabel(row.weekStart),
              count: row.count,
              buyAmount: row.buyAmount,
              sellAmount: row.sellAmount,
              isPartial: row.isPartial,
            }))}
          />
          <details className="mt-2 text-sm text-foreground/60">
            <summary className="cursor-pointer">数値を表で見る</summary>
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
          </details>
          <p className="mt-2 text-xs text-foreground/50">
            ※推定金額は発行済株式数×株価×保有比率の変化幅から概算した参考値です。
          </p>
        </section>
      )}

      {(buyCategoryRows.length > 0 || sellCategoryRows.length > 0) && (
        <section className="mb-10 border-y border-rule py-6">
          <h2 className="text-xl font-bold text-brand-navy">投資家分類別の週次トレンド</h2>
          <p className="mb-4 mt-2 text-sm leading-relaxed text-foreground/60">
            アクティビスト・事業会社・外資系運用会社といった提出者のタイプ別に、週ごとの
            開示件数と推定取引金額を並べたものです。買いと売りは打ち消し合わないよう別の表にしているので、
            どの種類の投資家がいま買い集めているのか・降りているのかが分かります。
          </p>

          {buyCategoryRows.length > 0 && (
            <>
              <h3 className="mt-6 text-base font-bold text-brand-navy">買い</h3>
              {latestBuyLeaders.length > 0 && (
                <p className="mt-1 text-sm leading-relaxed text-foreground/70">
                  直近の週（{latestWeek.tableLabel}
                  {latestWeek.isPartial && "・集計中"}）に最も買っていたのは
                  {latestBuyLeaders.map((c, i) => (
                    <span key={c.dealType}>
                      {i > 0 && "、"}
                      <span className="font-bold text-brand-navy">{c.dealType}</span>
                      （{c.count}件・{formatDealAmount(c.amount)}）
                    </span>
                  ))}
                  でした。
                </p>
              )}
              <CategoryWeeklyTrend rows={buyCategoryRows} weeks={categoryWeeks} tone="buy" />
            </>
          )}

          {sellCategoryRows.length > 0 && (
            <>
              <h3 className="mt-6 text-base font-bold text-brand-navy">売り</h3>
              {latestSellLeaders.length > 0 && (
                <p className="mt-1 text-sm leading-relaxed text-foreground/70">
                  直近の週（{latestWeek.tableLabel}
                  {latestWeek.isPartial && "・集計中"}）に最も売っていたのは
                  {latestSellLeaders.map((c, i) => (
                    <span key={c.dealType}>
                      {i > 0 && "、"}
                      <span className="font-bold text-brand-navy">{c.dealType}</span>
                      （{c.count}件・{formatDealAmount(c.amount)}）
                    </span>
                  ))}
                  でした。
                </p>
              )}
              <CategoryWeeklyTrend rows={sellCategoryRows} weeks={categoryWeeks} tone="sell" />
            </>
          )}

          <p className="mt-2 text-xs text-foreground/50">
            ※推定金額は発行済株式数×株価×保有比率の変化幅から概算した参考値です。
          </p>
        </section>
      )}

      <AdUnit placement="bottom" />
    </div>
  );
}
