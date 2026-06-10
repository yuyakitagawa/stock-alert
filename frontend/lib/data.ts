import type { Ranking, StockMeta, Earnings, AiAnalysis, CompanyProfile, QuarterlyEarning, Article, WeeklyReview, Activity, DividendCandidate } from "./types";
import { anonHeaders, sbUrl } from "./supabase";
import { yfQuoteSummary } from "./yahoo";

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

export async function fetchArticles(limit = 20): Promise<Article[]> {
  try {
    const res = await fetch(
      sbUrl(`articles?order=signal_date.desc&limit=${limit}`),
      { headers: anonHeaders(), next: { revalidate: 3600 } },
    );
    if (!res.ok) return [];
    return res.json();
  } catch {
    // ビルド時に env 未設定/URL不正で fetch が throw しても [] を返す
    return [];
  }
}

export async function fetchArticle(slug: string): Promise<Article | null> {
  try {
    const res = await fetch(
      sbUrl(`articles?slug=eq.${encodeURIComponent(slug)}&limit=1`),
      { headers: anonHeaders(), next: { revalidate: 3600 } },
    );
    if (!res.ok) return null;
    const rows = await res.json();
    return (rows[0] as Article) ?? null;
  } catch {
    return null;
  }
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

export async function fetchDividendCandidates(): Promise<DividendCandidate[]> {
  const res = await fetch(
    sbUrl("web_dividend_strategy?order=div_yield.desc&limit=30"),
    { headers: anonHeaders(), next: { revalidate: 300 } },
  );
  if (!res.ok) return [];
  const rows = await res.json();
  // 最新日付のデータのみ返す
  if (!rows.length) return [];
  const latest = rows[0].date;
  return (rows as DividendCandidate[]).filter(r => r.date === latest);
}
