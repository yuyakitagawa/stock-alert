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
  totalReturnOku: number;
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

// microCMSの`articles`スキーマには提出者名(filerName)フィールドが存在せず、
// web/publish_blog_articles.py がpayloadに載せている値はAPI側で黙って捨てられている
// （2026-08-15にAPIレスポンスのキー一覧で確認。CMSのスキーマ変更はGUIでしか行えないため
// コード側からは追加できない）。そのため記事から提出者を引くには、EDINET開示そのものを
// 持つSupabase側と「銘柄コード×開示日」で突き合わせるしかない。
// 同じ銘柄・同じ日に複数の提出者が開示しているケース（2026年8月実測で開示行の14%）は
// どの記事がどの提出者のものか一意に定まらないため、誤った帰属を出さないよう除外する。
export async function getFilerNamesByStockAndDate(
  from: string,
  to: string
): Promise<Map<string, string>> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("edinet_large_holdings")
    .select("issuer_code, disc_date, filer_name")
    .gte("disc_date", from)
    .lte("disc_date", to);

  const filersByKey = new Map<string, Set<string>>();
  for (const row of data ?? []) {
    if (!row.issuer_code || !row.disc_date || !row.filer_name) continue;
    const key = `${row.issuer_code}|${row.disc_date}`;
    const filers = filersByKey.get(key) ?? new Set<string>();
    filers.add(row.filer_name);
    filersByKey.set(key, filers);
  }

  const resolved = new Map<string, string>();
  for (const [key, filers] of filersByKey) {
    if (filers.size === 1) resolved.set(key, [...filers][0]);
  }
  return resolved;
}

// /ranking用。tools/filer_win_rate.pyが週次で再計算するfiler_win_rateテーブルを
// トータルリターン(total_return_oku)の降順で返す。
export async function getFilerWinRates(minN = 1): Promise<FilerWinRate[]> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("filer_win_rate")
    .select("filer_name, category, n, total_return_oku, hold_days, updated_at")
    .gte("n", minN)
    .order("total_return_oku", { ascending: false });

  return (data ?? []).map((r) => ({
    filerName: r.filer_name,
    category: r.category as DealType,
    n: r.n,
    totalReturnOku: r.total_return_oku,
    holdDays: r.hold_days,
    updatedAt: r.updated_at,
  }));
}
