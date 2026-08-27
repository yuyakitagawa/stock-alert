import type { HoldingRow } from "./investors";

// /trending（取引が急増した銘柄）用の期間比較。
// 記事(microCMS)ではなくEDINET開示(Supabase `edinet_large_holdings`)を数える。
// 記事の蓄積は2026年7月に始まったばかりで前期間がほぼ空になり比較が成立しないのに対し、
// 開示データは1年分あるため、今日時点でも意味のある「前期間比」を出せるため。
// ランキングの軸は開示件数のまま。金額（推定売買金額）は件数に添えて規模を示すために持つ。
// EDINET開示そのものに取引金額は無く、Supabase側のマテリアライズドビュー
// edinet_holding_amounts が「保有比率の変化幅 × 発行済株式数 × 開示日終値」で概算する。
// 訂正報告書や株価・株式数が取れない開示は金額不明＝0億円として扱う（件数には入る）ため、
// 件数と金額は必ずしも比例しない。

// 銘柄一覧の売買方向の絞り込み。「both」は方向が判定できない開示（保有比率が動かない
// 訂正報告書など）も含めた全件で、絞り込みを入れる前と同じ集計になる。
export type TrendingDirection = "buy" | "sell" | "both";

export type TrendingCounts = {
  count: number;
  prevCount: number;
  delta: number;
  // 推定売買金額の合計（億円）。countと同じ期間ぶん。
  amount: number;
  prevAmount: number;
  // 直前期間に開示が1件も無かった＝この期間で初めてクジラが動いた対象。
  isNew: boolean;
};

// 買い・売り・両方の3通りの件数を1件にまとめて持つ。絞り込みの切り替えで
// 再取得が起きないよう、3通りとも最初からクライアントに渡すため。
export type DirectionalTrendingEntry = { key: string; label: string } & Record<
  TrendingDirection,
  TrendingCounts
>;

type Bucket = { count: number; prevCount: number; amount: number; prevAmount: number };

function emptyBucket(): Bucket {
  return { count: 0, prevCount: 0, amount: 0, prevAmount: 0 };
}

function emptyBuckets(): Record<TrendingDirection, Bucket> {
  return { buy: emptyBucket(), sell: emptyBucket(), both: emptyBucket() };
}

function addDisclosure(bucket: Bucket, isCurrent: boolean, amountOku: number) {
  if (isCurrent) {
    bucket.count += 1;
    bucket.amount += amountOku;
  } else {
    bucket.prevCount += 1;
    bucket.prevAmount += amountOku;
  }
}

function collect(
  rows: HoldingRow[],
  currentFrom: string,
  amountByDocId: Record<string, number>,
  keyOf: (row: HoldingRow) => string,
  labelOf: (row: HoldingRow) => string
) {
  const entries = new Map<string, { label: string; buckets: Record<TrendingDirection, Bucket> }>();
  for (const row of rows) {
    const key = keyOf(row);
    if (!key) continue;
    const entry = entries.get(key) ?? { label: labelOf(row), buckets: emptyBuckets() };
    const isCurrent = row.discDate >= currentFrom;
    const amountOku = amountByDocId[row.docId] ?? 0;
    addDisclosure(entry.buckets.both, isCurrent, amountOku);
    if (row.direction !== "flat") addDisclosure(entry.buckets[row.direction], isCurrent, amountOku);
    entries.set(key, entry);
  }
  return entries;
}

function toCounts(bucket: Bucket): TrendingCounts {
  return {
    count: bucket.count,
    prevCount: bucket.prevCount,
    delta: bucket.count - bucket.prevCount,
    // 億円未満は表示しないので、ここで丸めて浮動小数の誤差を持ち回さない。
    amount: Math.round(bucket.amount),
    prevAmount: Math.round(bucket.prevAmount),
    isNew: bucket.prevCount === 0,
  };
}

// 増加件数が同じなら直近期間の件数が多いほうを上に出し、それも同じなら推定売買金額の
// 大きいほうを上に出す。7日窓では大半の銘柄が「+1件」で並ぶため（30日窓では件数だけで
// ほぼ順位が決まっていた）、金額まで見ないと順位が実質ランダムになる。
function compareTrending(a: TrendingCounts, b: TrendingCounts): number {
  return b.delta - a.delta || b.count - a.count || b.amount - a.amount;
}

// 選んだ方向の件数を実体化して、増えたものだけを並べ替えて返す。
export function selectDirection<T extends DirectionalTrendingEntry>(
  entries: T[],
  direction: TrendingDirection
): (T & TrendingCounts)[] {
  return entries
    .map((entry) => ({ ...entry, ...entry[direction] }))
    .filter((entry) => entry.delta > 0)
    .sort(compareTrending);
}

// /trendingの銘柄一覧。件数制限なし＝直近7日で開示が増えた銘柄をすべて返す。
// 並べ替えは絞り込みを切り替えるクライアント側（selectDirection）で行うため、
// ここではどの方向でも増えていない銘柄を落とすだけにする。
export function buildTrendingIssuers(
  rows: HoldingRow[],
  currentFrom: string,
  amountByDocId: Record<string, number> = {}
): DirectionalTrendingEntry[] {
  // EDINETの企業名は同じ銘柄でも表記が揺れる（「玉井商船株式会社」/「玉井商船　株式会社」）ため、
  // 集計キーは証券コード、表示は最初に現れた表記＋コードにする。
  const entries = collect(
    rows,
    currentFrom,
    amountByDocId,
    (r) => r.issuerCode,
    (r) => `${r.issuerName}（${r.issuerCode}）`
  );

  return [...entries]
    .map(([key, entry]) => ({
      key,
      label: entry.label,
      buy: toCounts(entry.buckets.buy),
      sell: toCounts(entry.buckets.sell),
      both: toCounts(entry.buckets.both),
    }))
    .filter((entry) => entry.buy.delta > 0 || entry.sell.delta > 0 || entry.both.delta > 0);
}
