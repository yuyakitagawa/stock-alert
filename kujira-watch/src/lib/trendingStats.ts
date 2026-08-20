import type { HoldingRow } from "./investors";

// /trending（クジラの動きが急増した銘柄・投資家）用の期間比較。
// 記事(microCMS)ではなくEDINET開示(Supabase `edinet_large_holdings`)を数える。
// 記事の蓄積は2026年7月に始まったばかりで前期間がほぼ空になり比較が成立しないのに対し、
// 開示データは1年分あるため、今日時点でも意味のある「前期間比」を出せるため。
// 開示には推定取引金額が無いので、比較の軸は金額ではなく開示件数にしている。

// 銘柄一覧の売買方向の絞り込み。「both」は方向が判定できない開示（保有比率が動かない
// 訂正報告書など）も含めた全件で、絞り込みを入れる前と同じ集計になる。
export type TrendingDirection = "buy" | "sell" | "both";

export type TrendingCounts = {
  count: number;
  prevCount: number;
  delta: number;
  // 直前期間に開示が1件も無かった＝この期間で初めてクジラが動いた対象。
  isNew: boolean;
};

export type TrendingEntry = TrendingCounts & {
  key: string;
  label: string;
};

// 買い・売り・両方の3通りの件数を1件にまとめて持つ。絞り込みの切り替えで
// 再取得が起きないよう、3通りとも最初からクライアントに渡すため。
export type DirectionalTrendingEntry = { key: string; label: string } & Record<
  TrendingDirection,
  TrendingCounts
>;

type Bucket = { count: number; prevCount: number };

function emptyBuckets(): Record<TrendingDirection, Bucket> {
  return {
    buy: { count: 0, prevCount: 0 },
    sell: { count: 0, prevCount: 0 },
    both: { count: 0, prevCount: 0 },
  };
}

function collect(
  rows: HoldingRow[],
  currentFrom: string,
  keyOf: (row: HoldingRow) => string,
  labelOf: (row: HoldingRow) => string
) {
  const entries = new Map<string, { label: string; buckets: Record<TrendingDirection, Bucket> }>();
  for (const row of rows) {
    const key = keyOf(row);
    if (!key) continue;
    const entry = entries.get(key) ?? { label: labelOf(row), buckets: emptyBuckets() };
    const field = row.discDate >= currentFrom ? "count" : "prevCount";
    entry.buckets.both[field] += 1;
    if (row.direction !== "flat") entry.buckets[row.direction][field] += 1;
    entries.set(key, entry);
  }
  return entries;
}

function toCounts(bucket: Bucket): TrendingCounts {
  return {
    count: bucket.count,
    prevCount: bucket.prevCount,
    delta: bucket.count - bucket.prevCount,
    isNew: bucket.prevCount === 0,
  };
}

// 増加件数が同じなら直近期間の件数が多いほうを上に出す。
function compareTrending(a: TrendingCounts, b: TrendingCounts): number {
  return b.delta - a.delta || b.count - a.count;
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

// /trendingの銘柄一覧。件数制限なし＝直近30日で開示が増えた銘柄をすべて返す。
// 並べ替えは絞り込みを切り替えるクライアント側（selectDirection）で行うため、
// ここではどの方向でも増えていない銘柄を落とすだけにする。
export function buildTrendingIssuers(
  rows: HoldingRow[],
  currentFrom: string
): DirectionalTrendingEntry[] {
  // EDINETの企業名は同じ銘柄でも表記が揺れる（「玉井商船株式会社」/「玉井商船　株式会社」）ため、
  // 集計キーは証券コード、表示は最初に現れた表記＋コードにする。
  const entries = collect(
    rows,
    currentFrom,
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

// /ranking/trendingの投資家一覧。売買方向の絞り込みは銘柄一覧だけの機能なので、
// こちらは従来どおり全開示（both）の増加件数順に上位limit件を返す。
export function buildTrendingFilers(
  rows: HoldingRow[],
  currentFrom: string,
  limit: number
): TrendingEntry[] {
  const entries = collect(rows, currentFrom, (r) => r.filerName, (r) => r.filerName);

  return [...entries]
    .map(([key, entry]) => ({ key, label: entry.label, ...toCounts(entry.buckets.both) }))
    .filter((entry) => entry.delta > 0)
    .sort(compareTrending)
    .slice(0, limit);
}
