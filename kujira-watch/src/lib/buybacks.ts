import { unstable_cache } from "next/cache";
import { getSupabaseServerClient } from "@/lib/supabase";

// 自社株買い（TDnet適時開示）の履歴。/stocks/[code] の「自社株買い」セクション用。
// 開示一覧は ext_tdnet_disclosures（category='自社株買い'、トレーディングシステム側の
// tools/fetch_tdnet.py が日次取得）、取得枠（上限株数・上限金額・比率・期間）は
// tdnet_buybacks（tools/enrich_buybacks.py が原文PDFから抽出）で、同じ (code, disclosed_at, title)
// をキーに突き合わせる。取得枠が付くのは「決定」開示だけで、月次の取得状況報告には付かない。
//
// 2026-08-23まで、このページは見出しに「大量保有・自社株買い履歴」と書きながらEDINETの
// 大量保有報告書しか出しておらず、自社株買いはDBに溜まるだけでサイトに一切出ていなかった。

const SUPABASE_REVALIDATE_SECONDS = 3600;

export type BuybackKind = "決定" | "進捗" | "その他";

export type BuybackRow = {
  disclosedAt: string; // ISO
  title: string;
  kind: BuybackKind;
  docUrl: string | null;
  maxShares: number | null;
  maxAmountYen: number | null;
  ratioPct: number | null;
  periodFrom: string | null;
  periodTo: string | null;
  method: string | null;
  willCancel: boolean | null;
};

// lib/buyback.py の classify_buyback_title() と同じ規律（進捗語を含むものは決定語が混ざっていても進捗）。
const PROGRESS_RE = /取得状況|取得結果|取得終了|買付け?結果|買付状況|取得の完了|事後調整/;
const DECISION_RE = /決定|決議|取得枠|取得に係る事項|買付けに関する|買付に関する|ToSTNeT|ＴｏＳＴＮｅＴ|N-NET|Ｎ－ＮＥＴ/;

export function buybackKind(title: string): BuybackKind {
  if (PROGRESS_RE.test(title)) return "進捗";
  if (DECISION_RE.test(title)) return "決定";
  return "その他";
}

// やのしんAPIのリダイレクタ（rd.php?<url>）を外してTDnet原文PDFへ直接リンクする。
export function tdnetPdfUrl(docUrl: string | null): string | null {
  if (!docUrl) return null;
  const m = docUrl.match(/^https?:\/\/webapi\.yanoshin\.jp\/rd\.php\?(https?:\/\/.+)$/);
  return m ? m[1] : docUrl;
}

// 上限金額の表示（1,045,800,000円 → "10.5億円"、24,240,000円 → "0.2億円"）。
export function formatAmountOku(yen: number | null): string | null {
  if (yen === null || yen <= 0) return null;
  const oku = yen / 100_000_000;
  if (oku >= 100) return `${Math.round(oku).toLocaleString("ja-JP")}億円`;
  return `${oku.toFixed(1)}億円`;
}

async function getBuybacksByStockCodeUncached(stockCode: string): Promise<BuybackRow[]> {
  const supabase = getSupabaseServerClient();
  const [{ data: disclosures }, { data: facts }] = await Promise.all([
    supabase
      .from("ext_tdnet_disclosures")
      .select("disclosed_at, title, doc_url")
      .eq("code", stockCode)
      .eq("category", "自社株買い")
      .order("disclosed_at", { ascending: false })
      .limit(50),
    supabase
      .from("tdnet_buybacks")
      .select(
        "disclosed_at, title, doc_url, max_shares, max_amount_yen, ratio_pct, period_from, period_to, method, will_cancel"
      )
      .eq("code", stockCode),
  ]);

  const factByKey = new Map((facts ?? []).map((f) => [`${f.disclosed_at} ${f.title}`, f]));
  return (disclosures ?? []).map((d) => {
    const f = factByKey.get(`${d.disclosed_at} ${d.title}`);
    return {
      disclosedAt: d.disclosed_at,
      title: d.title,
      kind: buybackKind(d.title),
      docUrl: tdnetPdfUrl(f?.doc_url ?? d.doc_url),
      maxShares: f?.max_shares ?? null,
      maxAmountYen: f?.max_amount_yen ?? null,
      ratioPct: f?.ratio_pct === null || f?.ratio_pct === undefined ? null : Number(f.ratio_pct),
      periodFrom: f?.period_from ?? null,
      periodTo: f?.period_to ?? null,
      method: f?.method ?? null,
      willCancel: f?.will_cancel ?? null,
    };
  });
}

export const getBuybacksByStockCode = unstable_cache(
  getBuybacksByStockCodeUncached,
  ["getBuybacksByStockCode"],
  { revalidate: SUPABASE_REVALIDATE_SECONDS }
);

// ---------------------------------------------------------------- /buybacks（TOPタブ）用

export type BuybackDecision = BuybackRow & {
  code: string;
  stockName: string;
};

// 直近days日の自社株買い「決定」（取得枠が抽出できたもの）。銘柄名はjpx_stock_listから引く。
// /buybacks の上限金額ランキング・発行済比率ランキング・最新一覧が共用する。
async function getRecentBuybackDecisionsUncached(days: number): Promise<BuybackDecision[]> {
  const supabase = getSupabaseServerClient();
  const since = new Date(Date.now() - days * 86400_000).toISOString().slice(0, 19);
  const { data: facts } = await supabase
    .from("tdnet_buybacks")
    .select(
      "code, disclosed_at, title, doc_url, max_shares, max_amount_yen, ratio_pct, period_from, period_to, method, will_cancel"
    )
    .gte("disclosed_at", since)
    .eq("extract_ok", true)
    .order("disclosed_at", { ascending: false })
    .limit(300);
  const rows = facts ?? [];
  if (rows.length === 0) return [];
  const codes = [...new Set(rows.map((r) => r.code))];
  const { data: names } = await supabase.from("jpx_stock_list").select("code, name").in("code", codes);
  const nameByCode = new Map((names ?? []).map((n) => [n.code, n.name as string]));
  return rows.map((f) => ({
    code: f.code,
    stockName: nameByCode.get(f.code) ?? f.code,
    disclosedAt: f.disclosed_at,
    title: f.title,
    kind: buybackKind(f.title),
    docUrl: tdnetPdfUrl(f.doc_url),
    maxShares: f.max_shares ?? null,
    maxAmountYen: f.max_amount_yen ?? null,
    ratioPct: f.ratio_pct === null || f.ratio_pct === undefined ? null : Number(f.ratio_pct),
    periodFrom: f.period_from ?? null,
    periodTo: f.period_to ?? null,
    method: f.method ?? null,
    willCancel: f.will_cancel ?? null,
  }));
}

export const getRecentBuybackDecisions = unstable_cache(
  getRecentBuybackDecisionsUncached,
  ["getRecentBuybackDecisions"],
  { revalidate: SUPABASE_REVALIDATE_SECONDS }
);

export type MonthlyBuybackCount = { month: string; decisions: number; totalYen: number; isPartial: boolean };

// 月別の決定件数と上限合計。件数はext_tdnet_disclosures（タイトル分類）から、金額はtdnet_buybacksから。
// 取得枠の抽出を始めた2026-08-13より前の月は金額が0になるため、件数だけの月として表示側で扱う。
async function getMonthlyBuybackCountsUncached(): Promise<MonthlyBuybackCount[]> {
  const supabase = getSupabaseServerClient();
  const [{ data: disclosures }, { data: facts }] = await Promise.all([
    supabase
      .from("ext_tdnet_disclosures")
      .select("disclosed_at, title")
      .eq("category", "自社株買い")
      .order("disclosed_at", { ascending: false })
      .limit(3000),
    supabase.from("tdnet_buybacks").select("disclosed_at, max_amount_yen").eq("extract_ok", true).limit(3000),
  ]);
  const currentMonth = new Date(Date.now() + 9 * 3600_000).toISOString().slice(0, 7);
  const byMonth = new Map<string, MonthlyBuybackCount>();
  for (const d of disclosures ?? []) {
    if (buybackKind(d.title) !== "決定") continue;
    const month = d.disclosed_at.slice(0, 7);
    const cur = byMonth.get(month) ?? { month, decisions: 0, totalYen: 0, isPartial: month === currentMonth };
    cur.decisions += 1;
    byMonth.set(month, cur);
  }
  for (const f of facts ?? []) {
    const month = f.disclosed_at.slice(0, 7);
    const cur = byMonth.get(month);
    if (cur) cur.totalYen += f.max_amount_yen ?? 0;
  }
  return [...byMonth.values()].sort((a, b) => (a.month < b.month ? 1 : -1));
}

export const getMonthlyBuybackCounts = unstable_cache(getMonthlyBuybackCountsUncached, ["getMonthlyBuybackCounts"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});
