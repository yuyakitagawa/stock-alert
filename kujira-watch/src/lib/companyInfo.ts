import { getSupabaseServerClient } from "@/lib/supabase";

export type CompanyInfo = {
  sector: string | null;
  hasYutai: boolean;
  yutaiMonth: number | null;
  close: number | null;
  closeDate: string | null;
  per: number | null;
  pbr: number | null;
  pos52: number | null;
};

// jpx_stock_list(銘柄マスター)とgen_rankings(直近営業日の株価・指標)は
// トレーディングシステム側(stock-alertリポジトリルート)が日次で更新しているテーブル。
// drop_prob/recommendなど売買シグナル自体はstock-alert本体の価値なのでここでは表示しない。
// 記事本体(microCMS)の表示を止めたくないため、会社情報の取得失敗はnullで握りつぶし
// あくまで付加情報として扱う。
export async function getCompanyInfo(code: string): Promise<CompanyInfo | null> {
  try {
    const supabase = getSupabaseServerClient();

    const [{ data: meta }, { data: ranking }] = await Promise.all([
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
        .limit(1)
        .maybeSingle(),
    ]);

    if (!meta && !ranking) return null;

    return {
      sector: meta?.sector ?? null,
      hasYutai: meta?.has_yutai ?? false,
      yutaiMonth: meta?.yutai_month ?? null,
      close: ranking?.close ?? null,
      closeDate: ranking?.date ?? null,
      per: ranking?.per ?? null,
      pbr: ranking?.pbr ?? null,
      pos52: ranking?.pos52 ?? null,
    };
  } catch (error) {
    console.error(`[getCompanyInfo] code=${code} 取得失敗`, error);
    return null;
  }
}
