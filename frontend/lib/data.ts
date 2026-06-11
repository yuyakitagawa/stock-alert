import type { Ranking, StockMeta, Earnings, AiAnalysis, CompanyProfile, QuarterlyEarning, WeeklyReview, Activity, WatchMetrics } from "./types";
import { anonHeaders, sbUrl } from "./supabase";
import { yfQuoteSummary, yfQuoteSummaryWithAuth, getAuth } from "./yahoo";

const CACHE: RequestInit = { next: { revalidate: 300 } };

const RECOMMEND_REMAP: Record<string, string> = {
  "高値警戒": "方向感なし",
  "売り検討": "下降シグナル",
};
function remapRecommend(r: Ranking): Ranking {
  const mapped = RECOMMEND_REMAP[r.recommend];
  return mapped ? { ...r, recommend: mapped } : r;
}

export async function fetchLatestDate(): Promise<string | null> {
  const res = await fetch(
    sbUrl("web_rankings?select=date&order=date.desc&limit=1"),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
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
    const res = await fetch(
      sbUrl(`web_rankings?date=eq.${date}&order=rank.asc&limit=${pageSize}&offset=${offset}`),
      { headers: anonHeaders(), ...CACHE }
    );
    if (!res.ok) break;
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
  const res = await fetch(
    sbUrl(`web_rankings?date=eq.${date}&code=eq.${code}&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return rows[0] ? remapRecommend(rows[0] as Ranking) : null;
}

export async function fetchStockMeta(code: string): Promise<StockMeta | null> {
  const res = await fetch(
    sbUrl(`web_stock_meta?code=eq.${code}&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as StockMeta) ?? null;
}

export async function fetchEarnings(code: string): Promise<Earnings | null> {
  const res = await fetch(
    sbUrl(`web_earnings?code=eq.${code}&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as Earnings) ?? null;
}

export async function fetchAiAnalysis(code: string): Promise<AiAnalysis | null> {
  const res = await fetch(
    sbUrl(`ai_analyses?code=eq.${code}&order=date.desc&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as AiAnalysis) ?? null;
}

export async function fetchSectorMap(): Promise<Record<string, string>> {
  const all: { code: string; sector: string | null }[] = [];
  const pageSize = 1000;
  let offset = 0;
  for (;;) {
    const res = await fetch(
      sbUrl(`web_stock_meta?select=code,sector&limit=${pageSize}&offset=${offset}`),
      { headers: anonHeaders(), next: { revalidate: 86400 } }
    );
    if (!res.ok) break;
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
  try {
    const result = await yfQuoteSummary(code, "assetProfile");
    if (!result) return { description: null, website: null, employees: null };
    const p = result.assetProfile as Record<string, unknown> | undefined;
    return {
      description: (p?.longBusinessSummary as string) ?? null,
      website:     (p?.website as string) ?? null,
      employees:   (p?.fullTimeEmployees as number) ?? null,
    };
  } catch {
    return { description: null, website: null, employees: null };
  }
}

export async function fetchRecentEarnings(code: string): Promise<QuarterlyEarning[]> {
  try {
    const result = await yfQuoteSummary(code, "incomeStatementHistoryQuarterly");
    if (!result) return [];
    const history = (result.incomeStatementHistoryQuarterly as Record<string, unknown>)
      ?.incomeStatementHistory;
    if (!Array.isArray(history)) return [];
    return (history as Record<string, unknown>[]).slice(0, 4).map(q => ({
      period:    (q.endDate as Record<string, unknown>)?.fmt as string ?? "—",
      revenue:   (q.totalRevenue as Record<string, unknown>)?.raw as number ?? null,
      netIncome: (q.netIncome as Record<string, unknown>)?.raw as number ?? null,
    }));
  } catch {
    return [];
  }
}

export async function fetchDailyQuote(code: string): Promise<import("./types").DailyQuote | null> {
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${code}.T?range=5d&interval=1d`;
    const res = await fetch(url, {
      next: { revalidate: 3600 },
      headers: { "User-Agent": "Mozilla/5.0 (compatible; StockSignal/1.0)" },
    });
    if (!res.ok) return null;
    const data = await res.json();
    const result = data?.chart?.result?.[0];
    if (!result) return null;

    const meta = result.meta ?? {};
    const timestamps: number[] = result.timestamp ?? [];
    const q = result.indicators?.quote?.[0] ?? {};
    const n = timestamps.length - 1;
    if (n < 0) return null;

    const price = (meta.regularMarketPrice as number) ?? null;
    const prev  = (meta.chartPreviousClose as number) ?? null;

    return {
      date:             timestamps[n] ? new Date(timestamps[n] * 1000).toISOString().split("T")[0] : null,
      price,
      open:             (q.open?.[n]   as number) ?? null,
      high:             (q.high?.[n]   as number) ?? null,
      low:              (q.low?.[n]    as number) ?? null,
      close:            (q.close?.[n]  as number) ?? null,
      volume:           (q.volume?.[n] as number) ?? null,
      prevClose:        prev,
      change:           price != null && prev != null ? price - prev : null,
      changePct:        price != null && prev != null ? ((price - prev) / prev) * 100 : null,
      fiftyTwoWeekHigh: (meta.fiftyTwoWeekHigh as number) ?? null,
      fiftyTwoWeekLow:  (meta.fiftyTwoWeekLow  as number) ?? null,
    };
  } catch {
    return null;
  }
}


export async function fetchSparkline(code: string): Promise<number[]> {
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${code}.T?range=1mo&interval=1d`;
    const res = await fetch(url, {
      next: { revalidate: 3600 },
      headers: { "User-Agent": "Mozilla/5.0 (compatible; StockSignal/1.0)" },
    });
    if (!res.ok) return [];
    const data = await res.json();
    const closes: (number | null)[] = data?.chart?.result?.[0]?.indicators?.quote?.[0]?.close ?? [];
    return closes.filter((v): v is number => v !== null);
  } catch {
    return [];
  }
}

// 日経225（^N225）の直近 days 営業日リターン(%)。業種別の絶対リターン算出に使用。
// （web_rankings には日経比 rel20 しか無いため、絶対リターン = rel20 + 日経20日 で復元する）
export async function fetchNikkeiReturn(days = 20): Promise<number> {
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/%5EN225?range=3mo&interval=1d`;
    const res = await fetch(url, {
      next: { revalidate: 3600 },
      headers: { "User-Agent": "Mozilla/5.0 (compatible; StockSignal/1.0)" },
    });
    if (!res.ok) return 0;
    const data = await res.json();
    const closes: number[] = (data?.chart?.result?.[0]?.indicators?.quote?.[0]?.close ?? [])
      .filter((v: number | null): v is number => v !== null);
    if (closes.length <= days) return 0;
    const last = closes[closes.length - 1];
    const prev = closes[closes.length - 1 - days];
    return prev > 0 ? ((last - prev) / prev) * 100 : 0;
  } catch {
    return 0;
  }
}



// 値上げ力ウォッチリスト用: 52週高値からの下落率（お得度）＋ PER/PBR
// 認証(crumb)は1回だけ取得して全銘柄で使い回す。全fetchにタイムアウトを入れ、
// 失敗時は null を返して描画をブロックしない（Vercelタイムアウト回避）。
export async function fetchWatchMetrics(
  code: string,
  auth: import("./yahoo").YahooAuth | null,
): Promise<WatchMetrics> {
  // 1) chart API（認証不要・Vercelでも安定）から現在値＋52週高値＋ミニチャート用終値。
  //    range=1mo で meta（高値・現在値）と直近1ヶ月の終値を1コールで両取り。
  let price: number | null = null;
  let high52: number | null = null;
  let spark: number[] = [];
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${code}.T?range=1mo&interval=1d`;
    const res = await fetch(url, {
      next: { revalidate: 3600 },
      headers: { "User-Agent": "Mozilla/5.0 (compatible; StockSignal/1.0)" },
      signal: AbortSignal.timeout(8000),
    });
    if (res.ok) {
      const data = await res.json();
      const result = data?.chart?.result?.[0];
      const meta = result?.meta;
      price = (meta?.regularMarketPrice as number) ?? null;
      high52 = (meta?.fiftyTwoWeekHigh as number) ?? null;
      const closes: (number | null)[] = result?.indicators?.quote?.[0]?.close ?? [];
      spark = closes.filter((v): v is number => v != null);
    }
  } catch { /* graceful */ }

  const drawdownPct =
    price != null && high52 != null && high52 > 0
      ? ((price - high52) / high52) * 100
      : null;

  // 2) PER / PBR（quoteSummary・共有認証）
  let per: number | null = null;
  let pbr: number | null = null;
  if (auth) {
    try {
      const r = await yfQuoteSummaryWithAuth(code, "summaryDetail,defaultKeyStatistics", auth);
      const sd = r?.summaryDetail as Record<string, { raw?: number }> | undefined;
      const ks = r?.defaultKeyStatistics as Record<string, { raw?: number }> | undefined;
      per = sd?.trailingPE?.raw ?? null;
      pbr = ks?.priceToBook?.raw ?? null;
    } catch { /* graceful */ }
  }

  return { price, high52, drawdownPct, per, pbr, spark };
}

// 複数銘柄を並列取得（認証は1回だけ）
export async function fetchWatchMetricsMap(codes: string[]): Promise<Record<string, WatchMetrics>> {
  const auth = await getAuth();
  const results = await Promise.all(
    codes.map(c => fetchWatchMetrics(c, auth).catch((): WatchMetrics | null => null)),
  );
  const map: Record<string, WatchMetrics> = {};
  codes.forEach((c, i) => { const m = results[i]; if (m) map[c] = m; });
  return map;
}

export async function fetchWeeklyReviews(limit = 10): Promise<WeeklyReview[]> {
  const res = await fetch(
    sbUrl(`weekly_reviews?order=week.desc&limit=${limit}`),
    { headers: anonHeaders(), next: { revalidate: 3600 } },
  );
  if (!res.ok) return [];
  return res.json();
}

export async function fetchActivity(limit = 60): Promise<Activity[]> {
  const res = await fetch(
    sbUrl(`activity_log?order=ts.desc&limit=${limit}`),
    { headers: anonHeaders(), next: { revalidate: 30 } },
  );
  if (!res.ok) return [];
  return res.json();
}

export async function fetchWeeklyReview(week: string): Promise<WeeklyReview | null> {
  const res = await fetch(
    sbUrl(`weekly_reviews?week=eq.${encodeURIComponent(week)}&limit=1`),
    { headers: anonHeaders(), next: { revalidate: 3600 } },
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as WeeklyReview) ?? null;
}
