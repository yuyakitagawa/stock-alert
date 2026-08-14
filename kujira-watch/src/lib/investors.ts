import { getSupabaseServerClient } from "@/lib/supabase";
import type { DealType } from "@/types/article";

export type FilerHolding = {
  docId: string;
  issuerCode: string;
  issuerName: string;
  holdingRatio: number | null;
  holdingRatioPrior: number | null;
  discDate: string;
  docTypeCode: string;
};

export type FilerClassification = {
  category: DealType;
  isForeign: boolean;
  description: string | null;
  profile: string | null;
};

export type FilerSummary = {
  filerName: string;
  category: DealType;
  holdingCount: number;
  latestDiscDate: string;
};

export type FilerWinRate = {
  filerName: string;
  category: DealType;
  n: number;
  winRate: number;
  shrunkWinRate: number;
  avgReturn: number;
  bigWinRate: number;
  holdDays: number;
  updatedAt: string;
};

// web/publish_blog_articles.py の DOC_TYPE_LABELS と対応。
const DOC_TYPE_LABELS: Record<string, string> = {
  "350": "大量保有報告書",
  "360": "変更報告書（保有比率の変更）",
};

export function docTypeLabel(code: string): string {
  return DOC_TYPE_LABELS[code] ?? "大量保有関連報告書";
}

export async function getFilerClassification(
  filerName: string
): Promise<FilerClassification | null> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("edinet_filer_classification")
    .select("category, is_foreign, description, profile")
    .eq("filer_name", filerName)
    .maybeSingle();
  if (!data) return null;
  return {
    category: data.category as DealType,
    isForeign: data.is_foreign,
    description: data.description,
    profile: data.profile,
  };
}

export async function getFilerHoldings(filerName: string): Promise<FilerHolding[]> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("edinet_large_holdings")
    .select("doc_id, issuer_code, issuer_name, holding_ratio, holding_ratio_prior, disc_date, doc_type_code")
    .eq("filer_name", filerName)
    .order("disc_date", { ascending: false })
    .limit(200);

  return (data ?? []).map((r) => ({
    docId: r.doc_id,
    issuerCode: r.issuer_code,
    issuerName: r.issuer_name ?? r.issuer_code,
    holdingRatio: r.holding_ratio,
    holdingRatioPrior: r.holding_ratio_prior,
    discDate: r.disc_date,
    docTypeCode: r.doc_type_code,
  }));
}

// /investors（一覧）とサイトマップ用。edinet_filer_summary(Supabaseビュー)は
// filer_name単位で1行に集計済みのため1000行上限に掛からない。
export async function getAllFilers(): Promise<FilerSummary[]> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("edinet_filer_summary")
    .select("filer_name, category, holding_count, latest_disc_date")
    .order("latest_disc_date", { ascending: false });

  return (data ?? []).map((r) => ({
    filerName: r.filer_name,
    category: r.category as DealType,
    holdingCount: r.holding_count,
    latestDiscDate: r.latest_disc_date,
  }));
}

// /stocks/[code] からのクロスリンク用。この銘柄に大量保有報告書を提出したことがある
// 投資家一覧（名称+分類）を返す。
export async function getFilersByStockCode(
  stockCode: string
): Promise<{ filerName: string; category: DealType }[]> {
  const supabase = getSupabaseServerClient();
  const { data: holdings } = await supabase
    .from("edinet_large_holdings")
    .select("filer_name")
    .eq("issuer_code", stockCode);

  const filerNames = [...new Set((holdings ?? []).map((h) => h.filer_name).filter(Boolean))];
  if (filerNames.length === 0) return [];

  const { data: classifications } = await supabase
    .from("edinet_filer_classification")
    .select("filer_name, category")
    .in("filer_name", filerNames);

  const categoryByFiler = new Map(
    (classifications ?? []).map((c) => [c.filer_name, c.category as DealType])
  );
  return filerNames
    .map((filerName) => ({
      filerName,
      category: categoryByFiler.get(filerName) ?? ("その他" as DealType),
    }))
    .sort((a, b) => a.filerName.localeCompare(b.filerName, "ja"));
}

// /ranking用。tools/filer_win_rate.pyが週次で再計算するfiler_win_rateテーブルを
// 収縮後勝率(shrunk_win_rate)の降順で返す（サンプル数が少ない投資家ほど分類平均に
// 寄せた値。生の勝率だけで並べるとn=1で100%の投資家が上位に来てしまうため）。
export async function getFilerWinRates(minN = 1): Promise<FilerWinRate[]> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("filer_win_rate")
    .select("filer_name, category, n, win_rate, shrunk_win_rate, avg_return, big_win_rate, hold_days, updated_at")
    .gte("n", minN)
    .order("shrunk_win_rate", { ascending: false });

  return (data ?? []).map((r) => ({
    filerName: r.filer_name,
    category: r.category as DealType,
    n: r.n,
    winRate: r.win_rate,
    shrunkWinRate: r.shrunk_win_rate,
    avgReturn: r.avg_return,
    bigWinRate: r.big_win_rate,
    holdDays: r.hold_days,
    updatedAt: r.updated_at,
  }));
}
