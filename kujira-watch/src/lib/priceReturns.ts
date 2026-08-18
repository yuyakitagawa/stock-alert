import { getSupabaseServerClient } from "./supabase";

// 記事ページの「開示後の株価推移」用。yahoo_price_cache（トレーディングシステム側が
// 日次更新している終値キャッシュ）から、開示日(dealDate)を基準にした
// 基準終値・+1ヶ月(21営業日)・+3ヶ月(63営業日)・直近の終値を返す。

export type PriceSnapshot = { date: string; close: number };

export type PriceAfterDisclosureData = {
  base: PriceSnapshot;
  oneMonth: PriceSnapshot | null;
  threeMonths: PriceSnapshot | null;
  latest: PriceSnapshot | null;
};

const ONE_MONTH_TRADING_DAYS = 21;
const THREE_MONTHS_TRADING_DAYS = 63;

// 開示日から基準終値までに許容する暦日ギャップ。これより離れている場合は
// 「開示時点の株価」と呼べないためデータ無し扱いにする（価格キャッシュの収録開始前の
// 古い開示や、上場廃止・コード変更などのケース）。
const MAX_BASE_GAP_DAYS = 7;

function daysBetween(from: string, to: string): number {
  return Math.round((Date.parse(to) - Date.parse(from)) / (24 * 60 * 60 * 1000));
}

// 記事本体(microCMS)の表示を止めたくないため、取得失敗はnullで握りつぶし
// あくまで付加情報として扱う（getCompanyInfoと同じ方針）。
export async function getPriceAfterDisclosure(
  stockCode: string,
  dealDate: string
): Promise<PriceAfterDisclosureData | null> {
  try {
    const supabase = getSupabaseServerClient();
    const { data } = await supabase
      .from("yahoo_price_cache")
      .select("date, close")
      .eq("code", stockCode)
      .gte("date", dealDate)
      .order("date", { ascending: true });

    const rows = (data ?? []).filter((r): r is { date: string; close: number } => r.close !== null);
    if (rows.length === 0) return null;

    const base = rows[0];
    if (daysBetween(dealDate, base.date) > MAX_BASE_GAP_DAYS) return null;

    const oneMonth = rows[ONE_MONTH_TRADING_DAYS] ?? null;
    const threeMonths = rows[THREE_MONTHS_TRADING_DAYS] ?? null;
    const last = rows[rows.length - 1];
    // 直近の終値が基準・+1ヶ月・+3ヶ月のいずれかと同じ日なら重複表示しない。
    const latest =
      last.date !== base.date && last.date !== oneMonth?.date && last.date !== threeMonths?.date
        ? last
        : null;

    return { base, oneMonth, threeMonths, latest };
  } catch (error) {
    console.error(`[getPriceAfterDisclosure] code=${stockCode} 取得失敗`, error);
    return null;
  }
}
