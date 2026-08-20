import { unstable_cache } from "next/cache";
import { getSupabaseServerClient } from "@/lib/supabase";
import type { DealType } from "@/types/article";

// /ranking/returns（投資家リターンランキング）の読み取り。
// 集計はSupabaseのマテリアライズドビュー investor_returns_3m 側で完結している
// （定義とレビュー観点は supabase/create_investor_returns_3m.sql）。
// アプリ側で組み立てると開示×株価で数千回のクエリになるため、ここでは並べて返すだけにする。

export type InvestorReturnRow = {
  filerName: string;
  category: DealType;
  positionCount: number;
  avgReturn: number;
  medianReturn: number;
  winRate: number;
  avgExcessReturn: number | null;
  bestStockCode: string;
  bestStockName: string;
  bestReturn: number;
  latestBuyDate: string;
  firstBuyDate: string;
};

// 集計対象の最小開示件数（ビュー側のHAVINGと同じ値。表示文言に使う）。
export const MIN_POSITIONS = 3;
// 3ヶ月＝63営業日。表示文言に使う。
export const RETURN_TRADING_DAYS = 63;

async function getInvestorReturnsUncached(limit: number): Promise<InvestorReturnRow[]> {
  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase
    .from("investor_returns_3m")
    .select(
      "filer_name, category, position_count, avg_return, median_return, win_rate, avg_excess_return, best_stock_code, best_stock_name, best_return, first_buy_date, latest_buy_date"
    )
    .order("avg_return", { ascending: false })
    .order("position_count", { ascending: false })
    .limit(limit);

  // /investors/[filer]と同じ規律で、読み取り失敗は握りつぶさず投げる（0件を正常な
  // 「該当なし」として空ページに焼き付けない）。
  if (error) throw new Error(`getInvestorReturns failed: ${error.message}`);

  return (data ?? []).map((r) => ({
    filerName: r.filer_name,
    category: (r.category ?? "その他") as DealType,
    positionCount: r.position_count,
    avgReturn: Number(r.avg_return),
    medianReturn: Number(r.median_return),
    winRate: r.win_rate,
    avgExcessReturn: r.avg_excess_return === null ? null : Number(r.avg_excess_return),
    bestStockCode: r.best_stock_code,
    bestStockName: r.best_stock_name,
    bestReturn: Number(r.best_return),
    firstBuyDate: r.first_buy_date,
    latestBuyDate: r.latest_buy_date,
  }));
}

// ビューの再計算は日次バッチ（daily_alert.yml）なので1時間キャッシュで十分。
export const getInvestorReturns = unstable_cache(getInvestorReturnsUncached, ["getInvestorReturns"], {
  revalidate: 3600,
});

// ランキング全体の母数。「n人中の何位か」を説明するために使う。
async function getInvestorReturnsSummaryUncached(): Promise<{
  filerCount: number;
  latestBuyDate: string | null;
}> {
  const supabase = getSupabaseServerClient();
  const [{ count }, { data }] = await Promise.all([
    supabase.from("investor_returns_3m").select("filer_name", { count: "exact", head: true }),
    supabase
      .from("investor_returns_3m")
      .select("latest_buy_date")
      .order("latest_buy_date", { ascending: false })
      .limit(1)
      .maybeSingle(),
  ]);
  return { filerCount: count ?? 0, latestBuyDate: data?.latest_buy_date ?? null };
}

export const getInvestorReturnsSummary = unstable_cache(
  getInvestorReturnsSummaryUncached,
  ["getInvestorReturnsSummary"],
  { revalidate: 3600 }
);

// 「+25.2%」「-3.4%」。0は符号なし。
export function formatSignedPercent(value: number, digits = 1): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

// 日経平均との差はパーセントではなくポイント差（％pt）として書く。
export function formatSignedPoint(value: number, digits = 1): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}pt`;
}
