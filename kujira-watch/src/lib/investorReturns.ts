import { unstable_cache } from "next/cache";
import { getSupabaseServerClient } from "@/lib/supabase";
import type { DealType } from "@/types/article";

// /ranking/returns（投資家リターンランキング）と /investors/[filer]（投資家ページ）の読み取り。
// 集計はSupabaseのマテリアライズドビュー側で完結している
// （定義とレビュー観点は supabase/create_investor_returns_3m.sql）。
//   investor_returns_3m           … 投資家1人＝1行の集計。ランキングと投資家ページの成績パネル。
//   investor_return_positions_3m  … 買い開示1件＝1行の明細。投資家ページの開示テーブルの「3ヶ月後」列。
// ランキングの数字と投資家ページの数字は必ず同じ行から作る（別々に計算するとズレても気付けない）。
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

// 投資家ページの成績パネル用。ランキング1行ぶんに「198名中◯位」を添えて返す。
// 買い開示が3件に満たない投資家はビューに載っていないのでnull（成績を出さない）。
export type FilerReturnSummary = {
  row: InvestorReturnRow;
  rank: number;
  filerCount: number;
};

async function getFilerReturnSummaryUncached(filerName: string): Promise<FilerReturnSummary | null> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("investor_returns_3m")
    .select(
      "filer_name, category, position_count, avg_return, median_return, win_rate, avg_excess_return, best_stock_code, best_stock_name, best_return, first_buy_date, latest_buy_date"
    )
    .eq("filer_name", filerName)
    .maybeSingle();
  if (!data) return null;

  // 順位は「自分より平均リターンが高い投資家の数+1」。同率は同じ順位になる。
  const [{ count: better }, { count: total }] = await Promise.all([
    supabase
      .from("investor_returns_3m")
      .select("filer_name", { count: "exact", head: true })
      .gt("avg_return", data.avg_return),
    supabase.from("investor_returns_3m").select("filer_name", { count: "exact", head: true }),
  ]);

  return {
    row: {
      filerName: data.filer_name,
      category: (data.category ?? "その他") as DealType,
      positionCount: data.position_count,
      avgReturn: Number(data.avg_return),
      medianReturn: Number(data.median_return),
      winRate: data.win_rate,
      avgExcessReturn: data.avg_excess_return === null ? null : Number(data.avg_excess_return),
      bestStockCode: data.best_stock_code,
      bestStockName: data.best_stock_name,
      bestReturn: Number(data.best_return),
      firstBuyDate: data.first_buy_date,
      latestBuyDate: data.latest_buy_date,
    },
    rank: (better ?? 0) + 1,
    filerCount: total ?? 0,
  };
}

export const getFilerReturnSummary = unstable_cache(
  getFilerReturnSummaryUncached,
  ["getFilerReturnSummary"],
  { revalidate: 3600 }
);

// 投資家ページの開示テーブル用。書類ID(doc_id)→3ヶ月後リターン(%)の対応表。
// 売り開示・訂正報告書・まだ3ヶ月経っていない開示は明細ビューに無いのでキーごと存在しない
// （呼び出し側はその開示の列を空欄にする）。
// 突合キーが(銘柄コード, 開示日)ではなくdoc_idなのは、同じ銘柄・同じ日に変更報告書と訂正報告書が
// 並ぶことがあり（実例: 2026-03-18のアーカス×SHIFT）、訂正報告書の行にまで数字が付いてしまうため。
// doc_idなら「ビューが実際に数えた1行」だけに数字が出るので、買い判定のルールをTS側に
// 書き写す必要もない（二重定義はズレたときに誰も気付けない）。
// 戻り値がMapではなく素のオブジェクトなのは、unstable_cacheがJSONで保存するため。
// Mapを返すとキャッシュ往復で`{}`に化けて`.get is not a function`で落ちる。
export type FilerReturnPositions = Record<string, number>;

async function getFilerReturnPositionsUncached(filerName: string): Promise<FilerReturnPositions> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("investor_return_positions_3m")
    .select("doc_id, ret_3m")
    .eq("filer_name", filerName)
    .limit(500);
  return Object.fromEntries((data ?? []).map((r) => [r.doc_id, Number(r.ret_3m)]));
}

export const getFilerReturnPositions = unstable_cache(
  getFilerReturnPositionsUncached,
  ["getFilerReturnPositions"],
  { revalidate: 3600 }
);
