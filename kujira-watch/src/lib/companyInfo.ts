import { getSupabaseServerClient } from "@/lib/supabase";

export type PricePoint = { date: string; close: number };

export type CompanyInfo = {
  sector: string | null;
  hasYutai: boolean;
  yutaiMonth: number | null;
  close: number | null;
  closeDate: string | null;
  per: number | null;
  pbr: number | null;
  pos52: number | null;
  priceHistory: PricePoint[];
};

// 株価グラフ用に遡る営業日数(約4ヶ月分)。
const PRICE_HISTORY_DAYS = 90;

// jpx_stock_list(銘柄マスター)とgen_rankings(日次の株価・指標)は
// トレーディングシステム側(stock-alertリポジトリルート)が日次で更新しているテーブル。
// drop_prob/recommendなど売買シグナル自体はstock-alert本体の価値なのでここでは表示しない。
// 記事本体(microCMS)の表示を止めたくないため、会社情報の取得失敗はnullで握りつぶし
// あくまで付加情報として扱う。
export async function getCompanyInfo(code: string): Promise<CompanyInfo | null> {
  try {
    const supabase = getSupabaseServerClient();

    const [{ data: meta }, { data: rankingRows }] = await Promise.all([
      supabase
        .from("jpx_stock_list")
        .select("sector, has_yutai, yutai_month")
        .eq("code", code)
        .maybeSingle(),
      supabase
        .from("gen_rankings")
        .select("date, close, per, pbr, pos52")
        .eq("code", code)
        .order("date", { ascending: false })
        .limit(PRICE_HISTORY_DAYS),
    ]);

    const latest = rankingRows?.[0];
    if (!meta && !latest) return null;

    const priceHistory = (rankingRows ?? [])
      .filter((r) => r.close !== null)
      .map((r) => ({ date: r.date, close: Number(r.close) }))
      .reverse();

    return {
      sector: meta?.sector ?? null,
      hasYutai: meta?.has_yutai ?? false,
      yutaiMonth: meta?.yutai_month ?? null,
      close: latest?.close ?? null,
      closeDate: latest?.date ?? null,
      per: latest?.per ?? null,
      pbr: latest?.pbr ?? null,
      pos52: latest?.pos52 ?? null,
      priceHistory,
    };
  } catch (error) {
    console.error(`[getCompanyInfo] code=${code} 取得失敗`, error);
    return null;
  }
}
