import { anonHeaders, sbUrl } from "./supabase";
import { fetchLatestDate } from "./data";

const SHARES = 100;

async function fetchEarliestDate(): Promise<string> {
  const res = await fetch(
    sbUrl("web_rankings?select=date&order=date.asc&limit=1"),
    { headers: anonHeaders(), next: { revalidate: 300 } }
  );
  if (!res.ok) return "2026-01-01";
  const rows = await res.json();
  return rows[0]?.date ?? "2026-01-01";
}

interface RawRow {
  code: string;
  name: string;
  close: number;
  date: string;
  recommend?: string;
}

export interface SimPosition {
  code:          string;
  name:          string;
  buyDate:       string;
  buyPrice:      number;
  status:        "held" | "sold";
  sellDate?:     string;
  sellPrice?:    number;
  currentPrice?: number;
  currentSignal?:string;
  pnl:           number;
  pnlPct:        number;
}

export interface SimSummary {
  totalCost:          number;
  totalValue:         number;
  totalPnl:           number;
  totalPnlPct:        number;
  heldCount:          number;
  soldCount:          number;
  winCount:           number;
  since:              string;
  allCount:           number;
  allWinCount:        number;
  avgReturnPct:       number;
  maxGainPct:         number;
  maxLossPct:         number;
  compoundReturnPct:  number;
  annualizedReturnPct:number;
}

// 特定コードの買い日以降の最初の売りシグナルを取得
async function fetchFirstSell(code: string, afterDate: string): Promise<RawRow | null> {
  const sellRecs = ["方向感なし", "弱気シグナル", "下降シグナル"]
    .map(r => encodeURIComponent(r)).join(",");
  const res = await fetch(
    sbUrl(`web_rankings?code=eq.${code}&date=gt.${afterDate}&recommend=in.(${sellRecs})&order=date.asc&limit=1&select=code,close,date`),
    { headers: anonHeaders(), next: { revalidate: 300 } }
  );
  if (!res.ok) return null;
  const rows: RawRow[] = await res.json();
  return rows[0] ?? null;
}

export async function fetchSimulation(): Promise<{
  positions: SimPosition[];
  summary: SimSummary;
}> {
  const sBuyEnc = encodeURIComponent("S買い");

  const [buyRows, latestDate, since] = await Promise.all([
    (async () => {
      const res = await fetch(
        sbUrl(`web_rankings?recommend=eq.${sBuyEnc}&net=gte.17&net=lte.24&drop_prob=lt.4&vol=lte.25&order=date.asc&select=code,name,close,date`),
        { headers: anonHeaders(), next: { revalidate: 300 } }
      );
      if (!res.ok) return [] as RawRow[];
      return res.json() as Promise<RawRow[]>;
    })(),
    fetchLatestDate(),
    fetchEarliestDate(),
  ]);

  // 銘柄ごとの最初のS買いのみ
  const firstBuy = new Map<string, RawRow>();
  for (const row of buyRows) {
    if (!firstBuy.has(row.code)) firstBuy.set(row.code, row);
  }

  // 最新価格マップ
  let latestMap = new Map<string, RawRow>();
  if (latestDate) {
    const pageSize = 1000;
    let offset = 0;
    for (;;) {
      const res = await fetch(
        sbUrl(`web_rankings?date=eq.${latestDate}&select=code,close,recommend&order=rank.asc&limit=${pageSize}&offset=${offset}`),
        { headers: anonHeaders(), next: { revalidate: 300 } }
      );
      if (!res.ok) break;
      const rows: RawRow[] = await res.json();
      for (const r of rows) latestMap.set(r.code, r);
      if (rows.length < pageSize) break;
      offset += pageSize;
    }
  }

  // 銘柄ごとに最初の売りシグナルを並列取得
  const buyEntries = Array.from(firstBuy.entries());
  const sellResults = await Promise.all(
    buyEntries.map(([code, buy]) => fetchFirstSell(code, buy.date))
  );
  const firstSell = new Map<string, RawRow | null>(
    buyEntries.map(([code], i) => [code, sellResults[i]])
  );

  const positions: SimPosition[] = buyEntries
    .sort(([, a], [, b]) => a.date.localeCompare(b.date))
    .map(([code, buy]) => {
      const sell = firstSell.get(code) ?? null;
      if (sell) {
        const pnl = (sell.close - buy.close) * SHARES;
        return {
          code,
          name:      buy.name,
          buyDate:   buy.date,
          buyPrice:  buy.close,
          status:    "sold" as const,
          sellDate:  sell.date,
          sellPrice: sell.close,
          pnl,
          pnlPct:    (sell.close - buy.close) / buy.close * 100,
        };
      } else {
        const latest = latestMap.get(code);
        const currentPrice = latest?.close ?? buy.close;
        const pnl = (currentPrice - buy.close) * SHARES;
        return {
          code,
          name:          buy.name,
          buyDate:       buy.date,
          buyPrice:      buy.close,
          status:        "held" as const,
          currentPrice,
          currentSignal: latest?.recommend,
          pnl,
          pnlPct:        (currentPrice - buy.close) / buy.close * 100,
        };
      }
    });

  const held = positions.filter(p => p.status === "held");
  const sold = positions.filter(p => p.status === "sold");

  // 全ポジション合算（保有中 + 売却済み）
  const allCost  = positions.reduce((s, p) => s + p.buyPrice * SHARES, 0);
  const allPnl   = positions.reduce((s, p) => s + p.pnl, 0);

  const allWinCount  = positions.filter(p => p.pnl > 0).length;
  const avgReturnPct = positions.length > 0
    ? positions.reduce((s, p) => s + p.pnlPct, 0) / positions.length
    : 0;
  const maxGainPct = positions.length > 0 ? Math.max(...positions.map(p => p.pnlPct)) : 0;
  const maxLossPct = positions.length > 0 ? Math.min(...positions.map(p => p.pnlPct)) : 0;

  // 複利リターン: 各トレードのリターンを順次複利で合算
  const compoundMult = positions.reduce((acc, p) => acc * (1 + p.pnlPct / 100), 1.0);
  const compoundReturnPct = (compoundMult - 1) * 100;

  // 年率換算: 観測期間(カレンダー日)から年率複利を計算
  const sinceMs   = new Date(since).getTime();
  const latestMs  = new Date(latestDate ?? since).getTime();
  const calDays   = Math.max(1, (latestMs - sinceMs) / (1000 * 60 * 60 * 24));
  const annualizedReturnPct = positions.length > 0
    ? (Math.pow(compoundMult, 365 / calDays) - 1) * 100
    : 0;

  return {
    positions,
    summary: {
      totalCost:           allCost,
      totalValue:          allCost + allPnl,
      totalPnl:            allPnl,
      totalPnlPct:         allCost > 0 ? (allPnl / allCost) * 100 : 0,
      heldCount:           held.length,
      soldCount:           sold.length,
      winCount:            held.filter(p => p.pnl > 0).length,
      since,
      allCount:            positions.length,
      allWinCount,
      avgReturnPct,
      maxGainPct,
      maxLossPct,
      compoundReturnPct,
      annualizedReturnPct,
    },
  };
}
