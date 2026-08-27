import type { Metadata } from "next";
import Link from "next/link";
import InfoTip from "@/components/InfoTip";
import { siblingDataPages } from "@/lib/nav";
import ListPageNextStep from "@/components/ListPageNextStep";
import Box from "@mui/material/Box";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import AmountTrendChart from "@/components/AmountTrendChart";
import CategoryTrendGrid, {
  type CategoryTrendRow,
  type CategoryWeekColumn,
} from "@/components/CategoryTrendGrid";
import RelatedArticles from "@/components/RelatedArticles";
import { getArticleList, getRecentArticleDigests, type ArticleDigest } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { formatDealAmount, isSellArticle } from "@/lib/format";
import type { DealType } from "@/types/article";
import AdUnit from "@/components/AdUnit";

// 「大口投資家の動きを教えて」等の包括的な検索・LLMクエリに直答するための集約ページ。
// 週ごとの推移（売買金額・投資家分類別）のグラフを主役にして「今が多いのか少ないのか・
// どの投資家が動いているか」のトレンドが分かるようにする（2026-08-18に週次トレンド化）。
// 冒頭にあった直近7日間の開示件数・推定取引金額タイルは、取得と売却を合算した規模で
// 方向が読めず、グラフと比べて伝える情報が無かったため2026-08-22に削除した。

// 週次トレンドの表示期間。記事データ（2026-07開始）から最大8週。
const AMOUNT_TREND_WEEKS = 8;

// 投資家分類別の週次トレンドは分類ごとの小さなグラフなので、週を増やすほど棒が細くなる。
// 「今どの分類が活発か」を読むのに必要な直近ぶんだけを、売買金額トレンドと同じ週区切りで出す。
const CATEGORY_TREND_WEEKS = 6;

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

// 今週を「集計中」として扱うか。EDINETの開示も記事生成（平日9〜21時JST）も平日にしか
// 動かないため、金曜の夜以降（土・日）は今週分の数字が確定している。暦週（月〜日）の
// 判定だけだと日曜まで「集計中」と出てしまい実態と合わない。
function isCurrentWeekPartial(today: string): boolean {
  const dow = new Date(`${today}T00:00:00Z`).getUTCDay();
  return dow !== 0 && dow !== 6;
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
  const today = todayJst();
  const currentWeekStart = weekStartOf(today);
  const currentWeekPartial = isCurrentWeekPartial(today);
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
      isPartial: weekStart === currentWeekStart && currentWeekPartial,
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
// グラフ同士で「同じ週」を見ていることが保証されるようにする。古い週→新しい週の並び。
// 買いと売りは打ち消し合うため合算せず、セル内で別々に持つ（分類ごとのグラフで上下に描く）。
function buildCategoryTrendRows(digests: ArticleDigest[], weekStarts: string[]): CategoryTrendRow[] {
  const weekIndex = new Map(weekStarts.map((weekStart, i) => [weekStart, i]));
  const byCategory = new Map<DealType, CategoryTrendRow>();
  for (const digest of digests) {
    const i = weekIndex.get(weekStartOf(digest.dealDate));
    if (i === undefined || !digest.dealType) continue;
    const row =
      byCategory.get(digest.dealType) ??
      {
        dealType: digest.dealType,
        cells: weekStarts.map(() => ({ buyCount: 0, buyAmount: 0, sellCount: 0, sellAmount: 0 })),
        buyTotal: 0,
        sellTotal: 0,
      };
    if (isSellArticle(digest.tags)) {
      row.cells[i].sellCount += 1;
      row.cells[i].sellAmount += digest.dealAmount;
      row.sellTotal += digest.dealAmount;
    } else {
      row.cells[i].buyCount += 1;
      row.cells[i].buyAmount += digest.dealAmount;
      row.buyTotal += digest.dealAmount;
    }
    byCategory.set(digest.dealType, row);
  }
  return [...byCategory.values()].sort((a, b) => b.buyTotal + b.sellTotal - (a.buyTotal + a.sellTotal));
}

export function generateMetadata(): Metadata {
  const title = "大口投資家の週次トレンド";
  const description =
    "EDINET大量保有報告書をもとにした大口投資家の週次トレンド。" +
    "週ごとの売買金額（買い/売り）と、アクティビスト・事業会社など投資家分類別の推移をグラフで確認できます。";
  const url = `${SITE_URL}/weekly`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function WeeklyDigestPage() {
  const [digests, { contents: latestArticles }] = await Promise.all([
    getRecentArticleDigests(AMOUNT_TREND_WEEKS * 7).catch(() => []),
    // グラフの下に添えるアイキャッチ付き記事カード用。取れなくてもページは成立させる。
    getArticleList({ limit: 20 }).catch(() => ({ contents: [] })),
  ]);
  const url = `${SITE_URL}/weekly`;

  // 直近の開示から推定取引金額の大きい取引（1銘柄1件まで）。グラフで「規模」を見た後に
  // 「中身」を読める導線として置く。
  const seenCodes = new Set<string>();
  const featuredArticles = [...latestArticles]
    .sort((a, b) => b.dealAmount - a.dealAmount)
    .filter((article) => {
      if (seenCodes.has(article.stockCode)) return false;
      seenCodes.add(article.stockCode);
      return true;
    })
    .slice(0, 4);

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
  const categoryRows = buildCategoryTrendRows(digests, categoryWeekStarts);

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
        <p className="mt-3 text-sm leading-relaxed text-foreground/70">
          大口投資家の週ごとの動きをグラフで掲載しています。
          <InfoTip content={`${SITE_NAME}がEDINET大量保有報告書をもとに集計。週別の売買金額（買い/売り）と投資家分類別の推移を示します。`} />
        </p>
      </div>

      {amountRows.length < 2 && (
        <p className="mb-10 text-sm leading-relaxed text-foreground/70">
          週次トレンドを表示できるだけの開示データがまだありません。
        </p>
      )}

      {amountRows.length >= 2 && (
        <section className="mb-10">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">週別の売買金額トレンド</h2>
          <p className="mb-2 text-sm text-foreground/60">
            上が買い、下が売り。上に大きく振れた週ほど買い越しです。
            <InfoTip content="解説記事化した開示の推定金額を週ごとに買い・売りへ分けた推移です。ベースラインより上が買い、下が売りです。" />
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

      {categoryRows.length > 0 && (
        <section className="mb-10 border-y border-rule py-6">
          <h2 className="text-xl font-bold text-brand-navy">投資家分類別の週次トレンド</h2>
          <p className="mb-2 mt-2 text-sm leading-relaxed text-foreground/60">
            アクティビスト・事業会社・外資系運用会社といった提出者のタイプ別に、週ごとの
            推定取引金額を買い（上）・売り（下）に分けて並べたものです。
            どの種類の投資家がいま買い集めているのか・降りているのかが分類ごとに分かります。
          </p>
          <CategoryTrendGrid rows={categoryRows} weeks={categoryWeeks} />
          <p className="mt-2 text-xs text-foreground/50">
            ※推定金額は発行済株式数×株価×保有比率の変化幅から概算した参考値です。
          </p>
        </section>
      )}

      <RelatedArticles
        title="直近の大型取引の解説記事"
        lead="直近の開示から推定取引金額の大きい取引をピックアップ。"
        articles={featuredArticles}
      />

      {/* データページ同士の横移動。ヘッダータブはあるが、GA4実測でTOPへの内部到達398件＝
          他ページからTOPへ戻る動きが多く、横に渡り歩けていなかった（2026-08-27）。 */}
      <ListPageNextStep links={siblingDataPages("/weekly")} />
      <AdUnit placement="bottom" />
    </div>
  );
}
