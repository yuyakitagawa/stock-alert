import type { Ranking, StockMeta, Earnings, AiAnalysis } from "./types";
import { anonHeaders, sbUrl } from "./supabase";

const CACHE: RequestInit = { next: { revalidate: 3600 } };

/** Fetch the latest date in web_rankings */
export async function fetchLatestDate(): Promise<string | null> {
  const res = await fetch(
    sbUrl("web_rankings?select=date&order=date.desc&limit=1"),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return rows[0]?.date ?? null;
}

/** Fetch all rankings for the latest date */
export async function fetchRankings(): Promise<{ date: string; rows: Ranking[] }> {
  const date = await fetchLatestDate();
  if (!date) return { date: "", rows: [] };

  const res = await fetch(
    sbUrl(`web_rankings?date=eq.${date}&order=rank.asc`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return { date, rows: [] };
  return { date, rows: (await res.json()) as Ranking[] };
}

/** Fetch a single stock ranking by code */
export async function fetchStockRanking(code: string): Promise<Ranking | null> {
  const date = await fetchLatestDate();
  if (!date) return null;

  const res = await fetch(
    sbUrl(`web_rankings?date=eq.${date}&code=eq.${code}&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as Ranking) ?? null;
}

/** Fetch stock metadata */
export async function fetchStockMeta(code: string): Promise<StockMeta | null> {
  const res = await fetch(
    sbUrl(`web_stock_meta?code=eq.${code}&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as StockMeta) ?? null;
}

/** Fetch earnings date */
export async function fetchEarnings(code: string): Promise<Earnings | null> {
  const res = await fetch(
    sbUrl(`web_earnings?code=eq.${code}&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as Earnings) ?? null;
}

/** Fetch AI analysis */
export async function fetchAiAnalysis(code: string): Promise<AiAnalysis | null> {
  const res = await fetch(
    sbUrl(`ai_analyses?code=eq.${code}&order=date.desc&limit=1`),
    { headers: anonHeaders(), ...CACHE }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0] as AiAnalysis) ?? null;
}
