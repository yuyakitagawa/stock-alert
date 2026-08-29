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

// 上場銘柄コードの全件セット。/stocks/[code] は解説記事が無くても
// 銘柄マスターに載っていれば開示履歴＋会社情報で成立するため、
// 一覧系ページ（/trending・/activists）が銘柄をリンクにしてよいかの
// 判定に使う（マスターに無い＝上場廃止等のコードだけテキストのまま出し、404リンクを作らない）。
const getAllListedCodeList = unstable_cache(
  async (): Promise<string[]> => {
    try {
      const supabase = getSupabaseServerClient();
      const codes: string[] = [];
      for (let from = 0; ; from += SUPABASE_PAGE_SIZE) {
        const { data } = await supabase
          .from("jpx_stock_list")
          .select("code")
          .range(from, from + SUPABASE_PAGE_SIZE - 1);
        for (const row of data ?? []) {
          if (row.code) codes.push(String(row.code));
        }
        if (!data || data.length < SUPABASE_PAGE_SIZE) break;
      }
      return codes;
    } catch (error) {
      console.error("[getAllListedCodeList] 取得失敗", error);
      return [];
    }
  },
  ["all-listed-codes"],
  { revalidate: 3600 }
);

export async function getAllListedCodes(): Promise<Set<string>> {
  return new Set(await getAllListedCodeList());
}

// 事業内容の説明（jpx_stock_list.description）が入っている銘柄コード。銘柄ページの
// インデックス判定（lib/pageIndexability.ts）に使う。判定はページ側とサイトマップ側の
// 両方で必要なので、全件を1度に取ってキャッシュする。
// unstable_cacheはJSONとして永続化するためSetではなく配列で保持する。
const getStockDescriptionCodeList = unstable_cache(
  async (): Promise<string[]> => {
    try {
      const supabase = getSupabaseServerClient();
      const codes: string[] = [];
      for (let from = 0; ; from += SUPABASE_PAGE_SIZE) {
        const { data } = await supabase
          .from("jpx_stock_list")
          .select("code, description")
          .not("description", "is", null)
          .neq("description", "")
          .range(from, from + SUPABASE_PAGE_SIZE - 1);
        for (const row of data ?? []) {
          if (row.code) codes.push(String(row.code));
        }
        if (!data || data.length < SUPABASE_PAGE_SIZE) break;
      }
      return codes;
    } catch (error) {
      console.error("[getStockDescriptionCodeList] 取得失敗", error);
      return [];
    }
  },
  ["stock-description-codes"],
  { revalidate: 3600 }
);

export async function getStockDescriptionCodes(): Promise<Set<string>> {
  return new Set(await getStockDescriptionCodeList());
}

export type CompanyInfo = {
  name: string | null;
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

    const [metaResult, rankingResult] = await Promise.all([
      supabase
        .from("jpx_stock_list")
        .select("name, sector, description, has_yutai, yutai_month")
        .eq("code", code)
        .maybeSingle(),
      supabase
        .from("gen_rankings")
        .select("date, close, per, pbr, pos52")
        .eq("code", code)
        .order("date", { ascending: false })
        .limit(PRICE_HISTORY_DAYS),
    ]);

    // supabase-jsは失敗を例外ではなく error で返すため、握り潰すと「原因がログに残らないまま
    // 銘柄ページが404になる」ことが起きる（2026-08-29、本番だけ47ページが404だった調査）。
    // 片方が落ちても、もう片方が取れていればページは成立させる。
    if (metaResult.error) {
      console.error(`[getCompanyInfo] code=${code} jpx_stock_list 取得失敗`, metaResult.error.message);
    }
    if (rankingResult.error) {
      console.error(`[getCompanyInfo] code=${code} gen_rankings 取得失敗`, rankingResult.error.message);
    }
    const meta = metaResult.data;
    const rankingRows = rankingResult.data;

    const latest = rankingRows?.[0];
    if (!meta && !latest) return null;

    const priceHistory = (rankingRows ?? [])
      .filter((r) => r.close !== null)
      .map((r) => ({ date: r.date, close: Number(r.close) }))
      .reverse();

    return {
      name: meta?.name ?? null,
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

// ヘッダー検索用の銘柄マスター検索。証券コードの前方一致 or 社名の部分一致で
// jpx_stock_list(全上場銘柄)を引く。microCMSの記事検索(searchStocks)だけだと
// 記事が1本も無い銘柄——EDINET開示はあるが記事化されていない約2,400銘柄——が
// 検索から完全に消えるため、マスター側でも引けるようにしている
// （実例: 6929 日本セラミックが検索に出ない、2026-08-18）。
const MASTER_SEARCH_LIMIT = 20;

export async function searchStockMaster(
  keyword: string
): Promise<{ stockCode: string; stockName: string }[]> {
  // PostgRESTのor句はカンマ・括弧で区切られるため、区切り文字とワイルドカードを除去する
  // （検索窓の自由入力でクエリが壊れる/意図しない全件マッチになるのを防ぐ）。
  const q = keyword.trim().replace(/[,()*%\\"']/g, "");
  if (!q) return [];

  try {
    const supabase = getSupabaseServerClient();
    const { data, error } = await supabase
      .from("jpx_stock_list")
      .select("code, name")
      .or(`code.ilike.${q}%,name.ilike.%${q}%`)
      .order("code")
      .limit(MASTER_SEARCH_LIMIT);
    if (error) {
      console.error(`[searchStockMaster] q=${keyword} 取得失敗`, error.message);
    }
    return (data ?? [])
      .filter((r) => r.code && r.name)
      .map((r) => ({ stockCode: String(r.code), stockName: r.name }));
  } catch (error) {
    console.error(`[searchStockMaster] q=${keyword} 取得失敗`, error);
    return [];
  }
}

// /trending の銘柄カードに「なんの会社か」を1行で添えるための一括取得。
// 事業内容(description)は約3割の銘柄にしか無いため、無い銘柄は業種(sector, ほぼ全銘柄にある)で代替する。
export type CompanyBrief = { description: string | null; sector: string | null };

export async function getCompanyBriefs(codes: string[]): Promise<Map<string, CompanyBrief>> {
  const briefs = new Map<string, CompanyBrief>();
  if (codes.length === 0) return briefs;

  try {
    const supabase = getSupabaseServerClient();
    const { data } = await supabase
      .from("jpx_stock_list")
      .select("code, description, sector")
      .in("code", codes);
    for (const row of data ?? []) {
      briefs.set(String(row.code), {
        description: row.description || null,
        sector: row.sector || null,
      });
    }
  } catch (error) {
    console.error("[getCompanyBriefs] 取得失敗", error);
  }
  return briefs;
}
