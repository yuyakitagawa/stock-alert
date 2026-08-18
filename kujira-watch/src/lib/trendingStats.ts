import type { HoldingRow } from "./investors";

// /trending（クジラの動きが急増した銘柄・投資家）用の期間比較。
// 記事(microCMS)ではなくEDINET開示(Supabase `edinet_large_holdings`)を数える。
// 記事の蓄積は2026年7月に始まったばかりで前期間がほぼ空になり比較が成立しないのに対し、
// 開示データは1年分あるため、今日時点でも意味のある「前期間比」を出せるため。
// 開示には推定取引金額が無いので、比較の軸は金額ではなく開示件数にしている。
export type TrendingEntry = {
  key: string;
  label: string;
  count: number;
  prevCount: number;
  delta: number;
  // 直前期間に開示が1件も無かった＝この期間で初めてクジラが動いた対象。
  isNew: boolean;
};

function buildTrending(
  rows: HoldingRow[],
  currentFrom: string,
  keyOf: (row: HoldingRow) => string,
  labelOf: (row: HoldingRow) => string,
  // 未指定なら全件返す（/trendingの銘柄一覧は直近30日で増えた銘柄をすべて出す）。
  limit?: number
): TrendingEntry[] {
  const entries = new Map<string, { label: string; count: number; prevCount: number }>();
  for (const row of rows) {
    const key = keyOf(row);
    if (!key) continue;
    const entry = entries.get(key) ?? { label: labelOf(row), count: 0, prevCount: 0 };
    if (row.discDate >= currentFrom) entry.count += 1;
    else entry.prevCount += 1;
    entries.set(key, entry);
  }

  return [...entries]
    .map(([key, entry]) => ({
      key,
      label: entry.label,
      count: entry.count,
      prevCount: entry.prevCount,
      delta: entry.count - entry.prevCount,
      isNew: entry.prevCount === 0,
    }))
    .filter((entry) => entry.delta > 0)
    // 増加件数が同じなら直近期間の件数が多いほうを上に出す。
    .sort((a, b) => b.delta - a.delta || b.count - a.count)
    .slice(0, limit);
}

export function buildTrendingIssuers(rows: HoldingRow[], currentFrom: string, limit?: number) {
  // EDINETの企業名は同じ銘柄でも表記が揺れる（「玉井商船株式会社」/「玉井商船　株式会社」）ため、
  // 集計キーは証券コード、表示は最初に現れた表記＋コードにする。
  return buildTrending(
    rows,
    currentFrom,
    (r) => r.issuerCode,
    (r) => `${r.issuerName}（${r.issuerCode}）`,
    limit
  );
}

export function buildTrendingFilers(rows: HoldingRow[], currentFrom: string, limit: number) {
  return buildTrending(rows, currentFrom, (r) => r.filerName, (r) => r.filerName, limit);
}
