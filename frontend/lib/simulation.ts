import { anonHeaders, sbUrl } from "./supabase";
import { fetchLatestDate } from "./data";

const SHARES = 100;
const SIM_START = "2026-01-01"; // シミュレーション開始日（固定）

interface QvRow {
  id:           number;
  code:         string;
  name:         string;
  entry_date:   string;
  exit_date:    string | null;
  entry_price:  number;
  exit_price:   number | null;
  return_pct:   number | null;
  reason:       string | null;
  held_days:    number | null;
  status:       string;
}

interface LatestRow {
  code:      string;
  close:     number;
  recommend: string;
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
  reason?:       string;
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

// QV戦略バックテスト結果（gen_qv_sim）から実績シミュレーションを構築する。
// QV戦略: Piotroski>=0.67 × pos52<0.45 × 業績改善、90日保有。
export async function fetchSimulation(): Promise<{
  positions: SimPosition[];
  summary: SimSummary;
}> {
  const [qvRows, latestDate] = await Promise.all([
    (async () => {
      const res = await fetch(
        sbUrl(`gen_qv_sim?order=entry_date.asc&limit=200`),
        { headers: anonHeaders(), cache: "no-store" }
      );
      if (!res.ok) return [] as QvRow[];
      return res.json() as Promise<QvRow[]>;
    })(),
    fetchLatestDate(),
  ]);

  // 保有中のコードに現在価格を取得
  const activeCodes = qvRows.filter(r => r.status === "active").map(r => r.code);
  let latestMap = new Map<string, LatestRow>();
  if (activeCodes.length > 0 && latestDate) {
    const inFilter = activeCodes.map(c => `code.eq.${c}`).join(",");
    const res = await fetch(
      sbUrl(`gen_rankings?date=eq.${latestDate}&or=(${inFilter})&select=code,close,recommend`),
      { headers: anonHeaders(), cache: "no-store" }
    );
    if (res.ok) {
      const rows: LatestRow[] = await res.json();
      for (const r of rows) latestMap.set(r.code, r);
    }
  }

  const positions: SimPosition[] = qvRows.map(row => {
    if (row.status === "active") {
      const latest  = latestMap.get(row.code);
      const curPrice = latest?.close ?? row.entry_price;
      const pnlPct   = (curPrice - row.entry_price) / row.entry_price * 100;
      return {
        code:          row.code,
        name:          row.name,
        buyDate:       row.entry_date,
        buyPrice:      row.entry_price,
        status:        "held" as const,
        currentPrice:  curPrice,
        currentSignal: latest?.recommend,
        pnl:           (curPrice - row.entry_price) * SHARES,
        pnlPct,
      };
    } else {
      const exitPx  = row.exit_price ?? row.entry_price;
      const pnlPct  = row.return_pct ?? (exitPx - row.entry_price) / row.entry_price * 100;
      return {
        code:      row.code,
        name:      row.name,
        buyDate:   row.entry_date,
        buyPrice:  row.entry_price,
        status:    "sold" as const,
        sellDate:  row.exit_date ?? undefined,
        sellPrice: exitPx,
        pnl:       (exitPx - row.entry_price) * SHARES,
        pnlPct,
        reason:    row.reason ?? undefined,
      };
    }
  });

  const held = positions.filter(p => p.status === "held");
  const sold = positions.filter(p => p.status === "sold");

  const allCost      = positions.reduce((s, p) => s + p.buyPrice * SHARES, 0);
  const allPnl       = positions.reduce((s, p) => s + p.pnl, 0);
  const allWinCount  = positions.filter(p => p.pnl > 0).length;
  const avgReturnPct = positions.length > 0
    ? positions.reduce((s, p) => s + p.pnlPct, 0) / positions.length : 0;
  const maxGainPct   = positions.length > 0 ? Math.max(...positions.map(p => p.pnlPct)) : 0;
  const maxLossPct   = positions.length > 0 ? Math.min(...positions.map(p => p.pnlPct)) : 0;

  const totalPnlPct       = allCost > 0 ? allPnl / allCost : 0;
  const compoundReturnPct = totalPnlPct * 100;

  const sinceMs  = new Date(SIM_START).getTime();
  const latestMs = new Date(latestDate ?? SIM_START).getTime();
  const calDays  = Math.max(1, (latestMs - sinceMs) / (1000 * 60 * 60 * 24));
  const annualizedReturnPct = positions.length > 0
    ? (Math.pow(1 + totalPnlPct, 365 / calDays) - 1) * 100 : 0;

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
      since:               SIM_START,
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
