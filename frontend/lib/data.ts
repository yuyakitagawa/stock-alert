import type { Ranking, StockMeta, Earnings, AiAnalysis, CompanyProfile, QuarterlyEarning, WatchMetrics, RiskRegime } from "./types";
import { anonHeaders, sbUrl } from "./supabase";

const CACHE: RequestInit = { next: { revalidate: 300 } };

// env欠落(NEXT_PUBLIC_SUPABASE_URL未設定で相対URL)やネットワーク失敗で fetch() 自体が
// throw し、ビルド(プリレンダ)が落ちるのを防ぐ共通ラッパ。失敗時は null を返す。
async function sbFetch(path: string, init: RequestInit): Promise<Response | null> {
  const url = sbUrl(path);
  // env欠落時 sbUrl は相対URL("/rest/v1/...")になる。Next.jsのfetchは相対URLを
  // throwせずハングするため try/catch では握れず、ビルド(プリレンダ)が60秒で
  // タイムアウトして落ちる。絶対URLでなければfetchせず即 null を返して回避する。
  if (!/^https?:\/\//.test(url)) return null;
  try {
    return await fetch(url, init);
  } catch {
    return null;
  }
}

const RECOMMEND_REMAP: Record<string, string> = {
  "高値警戒": "方向感なし",
};
function remapRecommend(r: Ranking): Ranking {
  const mapped = RECOMMEND_REMAP[r.recommend];
  return mapped ? { ...r, recommend: mapped } : r;
}

export async function fetchLatestDate(): Promise<string | null> {
  const res = await sbFetch("gen_rankings?select=date&order=date.desc&limit=1",
    { headers: anonHeaders(), ...CACHE });
  if (!res || !res.ok) return null;
  const rows = await res.json();
  return rows[0]?.date ?? null;
}

export async function fetchRankings(): Promise<{ date: string; rows: Ranking[] }> {
  const date = await fetchLatestDate();
  if (!date) return { date: "", rows: [] };

  const all: Ranking[] = [];
  const pageSize = 1000;
  let offset = 0;
  for (;;) {
    const res = await sbFetch(`gen_rankings?date=eq.${date}&order=rank.asc&limit=${pageSize}&offset=${offset}`,
      { headers: anonHeaders(), ...CACHE });
    if (!res || !res.ok) break;
    const rows: Ranking[] = await res.json();
    all.push(...rows);
    if (rows.length < pageSize) break;
    offset += pageSize;
  }
  return { date, rows: all.map(remapRecommend) };
}

export async function fetchStockRanking(code: string): Promise<Ranking | null> {
  const date = await fetchLatestDate();
  if (!date) return null;
  const res = await sbFetch(`gen_rankings?date=eq.${date}&code=eq.${code}&limit=1`,
    { headers: anonHeaders(), ...CACHE });
  if (!res || !res.ok) return null;
  const rows = await res.json();
  return rows[0] ? remapRecommend(rows[0] as Ranking) : null;
}

export async function fetchStockMeta(code: string): Promise<StockMeta | null> {
  const res = await sbFetch(`gen_stock_meta?code=eq.${code}&limit=1`,
    { headers: anonHeaders(), ...CACHE });
  if (!res || !res.ok) return null;
  const rows = await res.json();
  return (rows[0] as StockMeta) ?? null;
}

export async function fetchEarnings(code: string): Promise<Earnings | null> {
  const res = await sbFetch(`kabutan_earnings?code=eq.${code}&limit=1`,
    { headers: anonHeaders(), ...CACHE });
  if (!res || !res.ok) return null;
  const rows = await res.json();
  return (rows[0] as Earnings) ?? null;
}

export async function fetchAiAnalysis(code: string): Promise<AiAnalysis | null> {
  const res = await sbFetch(`gen_ai_analyses?code=eq.${code}&order=date.desc&limit=1`,
    { headers: anonHeaders(), ...CACHE });
  if (!res || !res.ok) return null;
  const rows = await res.json();
  return (rows[0] as AiAnalysis) ?? null;
}

export async function fetchSectorMap(): Promise<Record<string, string>> {
  const all: { code: string; sector: string | null }[] = [];
  const pageSize = 1000;
  let offset = 0;
  for (;;) {
    const res = await sbFetch(`gen_stock_meta?select=code,sector&limit=${pageSize}&offset=${offset}`,
      { headers: anonHeaders(), next: { revalidate: 86400 } });
    if (!res || !res.ok) break;
    const rows: { code: string; sector: string | null }[] = await res.json();
    all.push(...rows);
    if (rows.length < pageSize) break;
    offset += pageSize;
  }
  return Object.fromEntries(
    all.filter(r => r.sector).map(r => [r.code, r.sector!])
  );
}

export async function fetchCompanyProfile(code: string): Promise<CompanyProfile> {
  const res = await sbFetch(`gen_stock_meta?code=eq.${code}&limit=1&select=description,website,employees`,
    { headers: anonHeaders(), ...CACHE });
  if (!res || !res.ok) return { description: null, website: null, employees: null };
  const rows = await res.json();
  const r = rows[0] as Record<string, unknown> | undefined;
  return {
    description: (r?.description as string) ?? null,
    website:     (r?.website as string) ?? null,
    employees:   (r?.employees as number) ?? null,
  };
}

export async function fetchRecentEarnings(code: string): Promise<QuarterlyEarning[]> {
  const res = await sbFetch(
    `jquants_fin_summary?code=eq.${code}&order=fy_end.desc,disc_date.desc&limit=8&select=fy_end,doc_type,sales,np`,
    { headers: anonHeaders(), ...CACHE });
  if (!res || !res.ok) return [];
  const rows: { fy_end: string; doc_type: string; sales: number | null; np: number | null }[] = await res.json();
  const seen = new Set<string>();
  const unique: typeof rows = [];
  for (const r of rows) {
    const key = `${r.fy_end}_${r.doc_type}`;
    if (!seen.has(key)) { seen.add(key); unique.push(r); }
  }
  return unique.slice(0, 4).map(r => ({
    period:    `${r.fy_end} (${r.doc_type})`,
    revenue:   r.sales,
    netIncome: r.np,
  }));
}

export async function fetchDailyQuote(code: string): Promise<import("./types").DailyQuote | null> {
  try {
    const res = await sbFetch(
      `yahoo_price_cache?code=eq.${code}&order=date.desc&limit=5&select=date,close,volume`,
      { headers: anonHeaders(), next: { revalidate: 3600 } },
    );
    if (!res || !res.ok) return null;
    const rows: { date: string; close: number | null; volume: number | null }[] = await res.json();
    if (rows.length === 0) return null;

    const latest = rows[0];
    const prev = rows.length >= 2 ? rows[1] : null;
    const price = latest.close;
    const prevClose = prev?.close ?? null;

    return {
      date:             latest.date,
      price,
      open:             null,
      high:             null,
      low:              null,
      close:            price,
      volume:           latest.volume,
      prevClose,
      change:           price != null && prevClose != null ? price - prevClose : null,
      changePct:        price != null && prevClose != null ? ((price - prevClose) / prevClose) * 100 : null,
      fiftyTwoWeekHigh: null,
      fiftyTwoWeekLow:  null,
    };
  } catch {
    return null;
  }
}


export async function fetchSparkline(code: string): Promise<number[]> {
  try {
    const cutoff = new Date(Date.now() - 30 * 86400000).toISOString().split("T")[0];
    const res = await sbFetch(
      `yahoo_price_cache?code=eq.${code}&date=gte.${cutoff}&order=date.asc&select=close`,
      { headers: anonHeaders(), next: { revalidate: 3600 } },
    );
    if (!res || !res.ok) return [];
    const rows: { close: number | null }[] = await res.json();
    return rows.map(r => r.close).filter((v): v is number => v !== null);
  } catch {
    return [];
  }
}

// 日経225（^N225）の直近 days 営業日リターン(%)。業種別の絶対リターン算出に使用。
// （gen_rankings には日経比 rel20 しか無いため、絶対リターン = rel20 + 日経20日 で復元する）
export async function fetchNikkeiReturn(days = 20): Promise<number> {
  try {
    const cutoff = new Date(Date.now() - 90 * 86400000).toISOString().split("T")[0];
    const res = await sbFetch(
      `yahoo_market_index?ticker=eq.N225&date=gte.${cutoff}&order=date.asc&select=close`,
      { headers: anonHeaders(), next: { revalidate: 3600 } },
    );
    if (!res || !res.ok) return 0;
    const rows: { close: number | null }[] = await res.json();
    const closes: number[] = rows.map(r => r.close).filter((v): v is number => v !== null)
      .filter((v: number | null): v is number => v !== null);
    if (closes.length <= days) return 0;
    const last = closes[closes.length - 1];
    const prev = closes[closes.length - 1 - days];
    return prev > 0 ? ((last - prev) / prev) * 100 : 0;
  } catch {
    return 0;
  }
}




export async function fetchRiskRegime(): Promise<RiskRegime | null> {
  const res = await sbFetch("gen_risk_regime?order=date.desc&limit=1",
    { headers: anonHeaders(), next: { revalidate: 300 } });
  if (!res || !res.ok) return null;
  const rows = await res.json();
  return (rows[0] as RiskRegime) ?? null;
}

export async function fetchWatchMetrics(code: string): Promise<WatchMetrics> {
  let price: number | null = null;
  let high52: number | null = null;
  let spark: number[] = [];
  try {
    const cutoff1m = new Date(Date.now() - 30 * 86400000).toISOString().split("T")[0];
    const res1 = await sbFetch(
      `yahoo_price_cache?code=eq.${code}&date=gte.${cutoff1m}&order=date.asc&select=close`,
      { headers: anonHeaders(), next: { revalidate: 3600 } },
    );
    if (res1 && res1.ok) {
      const rows: { close: number | null }[] = await res1.json();
      spark = rows.map(r => r.close).filter((v): v is number => v != null);
      if (spark.length > 0) price = spark[spark.length - 1];
    }
    const cutoff1y = new Date(Date.now() - 365 * 86400000).toISOString().split("T")[0];
    const res2 = await sbFetch(
      `yahoo_price_cache?code=eq.${code}&date=gte.${cutoff1y}&order=date.asc&select=close`,
      { headers: anonHeaders(), next: { revalidate: 3600 } },
    );
    if (res2 && res2.ok) {
      const rows2: { close: number | null }[] = await res2.json();
      const closes52 = rows2.map(r => r.close).filter((v): v is number => v != null);
      high52 = closes52.length > 0 ? Math.max(...closes52) : null;
    }
  } catch { /* graceful */ }

  const drawdownPct =
    price != null && high52 != null && high52 > 0
      ? ((price - high52) / high52) * 100
      : null;

  let per: number | null = null;
  let pbr: number | null = null;
  try {
    const res3 = await sbFetch(
      `gen_stock_meta?code=eq.${code}&limit=1&select=per,pbr`,
      { headers: anonHeaders(), ...CACHE },
    );
    if (res3 && res3.ok) {
      const rows3 = await res3.json();
      if (rows3[0]) {
        per = rows3[0].per ?? null;
        pbr = rows3[0].pbr ?? null;
      }
    }
  } catch { /* graceful */ }

  return { price, high52, drawdownPct, per, pbr, spark };
}

export async function fetchWatchMetricsMap(codes: string[]): Promise<Record<string, WatchMetrics>> {
  const results = await Promise.all(
    codes.map(c => fetchWatchMetrics(c).catch((): WatchMetrics | null => null)),
  );
  const map: Record<string, WatchMetrics> = {};
  codes.forEach((c, i) => { const m = results[i]; if (m) map[c] = m; });
  return map;
}
