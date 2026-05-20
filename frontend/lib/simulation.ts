import { anonHeaders, sbUrl } from "./supabase";
import { fetchLatestDate } from "./data";

const SHARES = 100;
const PAGE = 1000;

async function fetchEarliestDate(): Promise<string> {
  const res = await fetch(
    sbUrl("web_rankings?select=date&order=date.asc&limit=1"),
    { headers: anonHeaders(), next: { revalidate: 3600 } }
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
  totalCost:    number;
  totalValue:   number;
  totalPnl:     number;
  totalPnlPct:  number;
  heldCount:    number;
  soldCount:    number;
  winCount:     number;
  since:        string;
  allCount:     number;
  allWinCount:  number;
  avgReturnPct: number;
  maxGainPct:   number;
  maxLossPct:   number;
}

async function fetchAll(path: string): Promise<RawRow[]> {
  const all: RawRow[] = [];
  let offset = 0;
  for (;;) {
    const res = await fetch(
      sbUrl(`${path}&limit=${PAGE}&offset=${offset}`),
      { headers: anonHeaders(), next: { revalidate: 3600 } }
    );
    if (!res.ok) break;
    const rows: RawRow[] = await res.json();
    all.push(...rows);
    if (rows.length < PAGE) break;
    offset += PAGE;
  }
  return all;
}

export async function fetchSimulation(): Promise<{
  positions: SimPosition[];
  summary: SimSummary;
}> {
  const sBuyEnc      = encodeURIComponent("S買い");
  const neutralEnc   = encodeURIComponent("方向感なし");
  const weakEnc      = encodeURIComponent("弱気シグナル");
  const downEnc      = encodeURIComponent("下降シグナル");

  const [buyRows, sellRows, latestDate, since] = await Promise.all([
    fetchAll(`web_rankings?recommend=eq.${sBuyEnc}&order=date.asc&select=code,name,close,date`),
    fetchAll(`web_rankings?recommend=in.(${neutralEnc},${weakEnc},${downEnc})&order=date.asc&select=code,close,date`),
    fetchLatestDate(),
    fetchEarliestDate(),
  ]);

  // Latest prices for held stocks
  let latestMap = new Map<string, RawRow>();
  if (latestDate) {
    const rows = await fetchAll(
      `web_rankings?date=eq.${latestDate}&select=code,close,recommend`
    );
    latestMap = new Map(rows.map(r => [r.code, r]));
  }

  // First S買い per code
  const firstBuy = new Map<string, RawRow>();
  for (const row of buyRows) {
    if (!firstBuy.has(row.code)) firstBuy.set(row.code, row);
  }

  // First 方向感なし/弱気/下降シグナル after buy date per code (net<6 = メール売り基準)
  const firstSell = new Map<string, RawRow>();
  for (const row of sellRows) {
    const buy = firstBuy.get(row.code);
    if (buy && row.date > buy.date && !firstSell.has(row.code)) {
      firstSell.set(row.code, row);
    }
  }

  const positions: SimPosition[] = Array.from(firstBuy.values())
    .sort((a, b) => a.date.localeCompare(b.date))
    .map(buy => {
      const sell = firstSell.get(buy.code);
      if (sell) {
        const pnl = (sell.close - buy.close) * SHARES;
        return {
          code:      buy.code,
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
        const latest = latestMap.get(buy.code);
        const currentPrice = latest?.close ?? buy.close;
        const pnl = (currentPrice - buy.close) * SHARES;
        return {
          code:          buy.code,
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
  const totalCost  = held.reduce((s, p) => s + p.buyPrice * SHARES, 0);
  const totalValue = held.reduce((s, p) => s + (p.currentPrice ?? p.buyPrice) * SHARES, 0);
  const totalPnl   = totalValue - totalCost;

  const allWinCount  = positions.filter(p => p.pnl > 0).length;
  const avgReturnPct = positions.length > 0
    ? positions.reduce((s, p) => s + p.pnlPct, 0) / positions.length
    : 0;
  const maxGainPct = positions.length > 0 ? Math.max(...positions.map(p => p.pnlPct)) : 0;
  const maxLossPct = positions.length > 0 ? Math.min(...positions.map(p => p.pnlPct)) : 0;

  return {
    positions,
    summary: {
      totalCost,
      totalValue,
      totalPnl,
      totalPnlPct: totalCost > 0 ? (totalPnl / totalCost) * 100 : 0,
      heldCount:   held.length,
      soldCount:   sold.length,
      winCount:    held.filter(p => p.pnl > 0).length,
      since,
      allCount:     positions.length,
      allWinCount,
      avgReturnPct,
      maxGainPct,
      maxLossPct,
    },
  };
}
