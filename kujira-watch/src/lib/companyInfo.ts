import { unstable_cache } from "next/cache";
import { getSupabaseServerClient } from "@/lib/supabase";

export type PricePoint = { date: string; close: number };

// 銘柄一覧ページの業種フィルター用。jpx_stock_list(証券コードのマスター、日次更新)を
// 全件取得してMapにする。/stocksが記事一覧の取得(getAllStocksForIndex)と並列実行できるよう、
// 記事側のstockCodeに依存しない全件取得にしている。マスターデータで更新頻度が低いため
// 記事側(REVALIDATE_SECONDS=60)より長く1時間キャッシュする。
const SUPABASE_PAGE_SIZE = 1000;

// unstable_cacheはNext.jsのData CacheにJSONとして永続化するため、Mapを直接返すと
// キャッシュヒット時にプレーンオブジェクトへ化けて.get()が壊れる。
// [code, sector]の配列でキャッシュし、呼び出し側でMapへ復元する。
const getAllSectorEntries = unstable_cache(
  async (): Promise<[string, string][]> => {
    try {
      const supabase = getSupabaseServerClient();
      const entries: [string, string][] = [];

      // jpx_stock_list(約4,500銘柄)はSupabase/PostgRESTのデフォルト行数上限(1,000件)を
      // 超えるため、.range()で全件を取り切るまでページングする（上限に気づかず
      // 一部銘柄の業種が欠落するバグを防ぐ）。
      for (let from = 0; ; from += SUPABASE_PAGE_SIZE) {
        const { data } = await supabase
          .from("jpx_stock_list")
          .select("code, sector")
          .range(from, from + SUPABASE_PAGE_SIZE - 1);
        for (const row of data ?? []) {
          if (row.sector) entries.push([row.code, row.sector]);
        }
        if (!data || data.length < SUPABASE_PAGE_SIZE) break;
      }
      return entries;
    } catch (error) {
      console.error("[getAllSectorEntries] 取得失敗", error);
      return [];
    }
  },
  ["all-sectors-by-code"],
  { revalidate: 3600 }
);

export async function getAllSectorsByCode(): Promise<Map<string, string>> {
  return new Map(await getAllSectorEntries());
}

export type CompanyInfo = {
  sector: string | null;
  description: string | null;
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
        .select("sector, description, has_yutai, yutai_month")
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
      description: meta?.description ?? null,
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
