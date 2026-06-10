import { anonHeaders, sbUrl } from "./supabase";
import { fetchLatestDate } from "./data";

const SHARES = 100;
const SELL_NET_THRESH = 5;   // ネットスコアがこれ未満に下がったら売却（信号消滅）

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
  net?: number;
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

// 特定コードの買い日以降、ネットスコアが閾値未満に下がった最初の日を売却日とする
async function fetchFirstSell(code: string, afterDate: string): Promise<RawRow | null> {
  const res = await fetch(
    sbUrl(`web_rankings?code=eq.${code}&date=gt.${afterDate}&net=lt.${SELL_NET_THRESH}&order=date.asc&limit=1&select=code,close,date,net`),
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
  // 毎日のネットスコア上位10銘柄（rank<=10）を買い候補とする。
  // rank は export 時に net 降順で振られた日次順位（1 = 最高ネット）。
  const [buyRows, latestDate, since] = await Promise.all([
    (async () => {
      const all: RawRow[] = [];
      const pageSize = 1000;
      let offset = 0;
      for (;;) {
        const res = await fetch(
          sbUrl(`web_rankings?rank=lte.10&order=date.asc,rank.asc&select=code,name,close,date,net&limit=${pageSize}&offset=${offset}`),
          { headers: anonHeaders(), next: { revalidate: 300 } }
        );
        if (!res.ok) break;
        const rows: RawRow[] = await res.json();
        all.push(...rows);
        if (rows.length < pageSize) break;
        offset += pageSize;
      }
      return all;
    })(),
    fetchLatestDate(),
    fetchEarliestDate(),
  ]);

  // 銘柄ごとに「最初にtop10入りした日」を買い日とする
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

  // 期間リターン: 総損益 ÷ 総投資額（独立トレードの合算として正確）
  const totalPnlPct = allCost > 0 ? allPnl / allCost : 0;
  const compoundReturnPct = totalPnlPct * 100;

  // 年率換算: 期間リターンを観測期間のカレンダー日数で年率換算
  const sinceMs   = new Date(since).getTime();
  const latestMs  = new Date(latestDate ?? since).getTime();
  const calDays   = Math.max(1, (latestMs - sinceMs) / (1000 * 60 * 60 * 24));
  const annualizedReturnPct = positions.length > 0
    ? (Math.pow(1 + totalPnlPct, 365 / calDays) - 1) * 100
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
