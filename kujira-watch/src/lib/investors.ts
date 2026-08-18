import { unstable_cache } from "next/cache";

// EDINET開示(Supabase)の読み取りは全てunstable_cacheに載せる。
// supabase-jsのfetchはキャッシュ指定が無く、Next.js 15以降のfetchは既定でno-storeのため、
// 素で呼ぶとそのページ全体が動的レンダリング扱いになる（実測: /articles/* も /stocks/* も
// x-vercel-cache: MISS・cache-control: no-store で毎回サーバー実行、TTFB約2秒）。
// クローラーは同じURLを何度も取りに来るので、これがそのままクロール速度の上限になっていた。
// EDINET開示は日次更新なので1時間のキャッシュで十分。
import { getSupabaseServerClient } from "@/lib/supabase";
import type { DealType } from "@/types/article";

// PostgRESTは1リクエストあたり既定1000行で打ち切り、しかも返る行の順序も保証されない
// （実測: 開示60日間で2,906行、投資家一覧で2,938行）。全件必要なクエリは並び順を固定した
// うえでこの単位でページングして取り切る。
// Supabase読み取りのキャッシュ保持時間。EDINET開示は日次更新なので1時間で十分。
const SUPABASE_REVALIDATE_SECONDS = 3600;

const PAGE_SIZE = 1000;
// 暴走防止の上限（20ページ＝2万行）。
const MAX_PAGES = 20;

export type FilerHolding = {
  docId: string;
  issuerCode: string;
  issuerName: string;
  holdingRatio: number | null;
  holdingRatioPrior: number | null;
  discDate: string;
  docTypeCode: string;
  docDescription: string | null;
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

async function getFilerClassificationUncached(
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

export const getFilerClassification = unstable_cache(getFilerClassificationUncached, ["getFilerClassification"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

// 呼び出し元(/investors/[filer])は戻り値0件をそのまま notFound() に繋いでいる。
// ここでSupabaseの読み取りエラーを握りつぶして[]を返すと、一時的な障害が「存在しない投資家」として
// 404ページに焼き付き、ISRキャッシュに残ったままGooglebotに配信される（実測: 2026-08-19時点で
// 主要投資家ページの大半が初回アクセス404・2回目以降200という状態になっていた）。
// 取得失敗は必ず例外にして、Nextに「生成失敗」として扱わせる（直前の正常なキャッシュが使われる）。
async function getFilerHoldingsUncached(filerName: string): Promise<FilerHolding[]> {
  const supabase = getSupabaseServerClient();
  const query = () =>
    supabase
      .from("edinet_large_holdings")
      .select("doc_id, issuer_code, issuer_name, holding_ratio, holding_ratio_prior, disc_date, doc_type_code, doc_description")
      .eq("filer_name", filerName)
      .order("disc_date", { ascending: false })
      .limit(200);

  // ビルド時は100ページ分の事前生成が同時にSupabaseを叩くため、一過性の失敗で例外にすると
  // デプロイ全体が落ちる。1度だけ再試行し、それでも駄目なときだけ投げる。
  let { data, error } = await query();
  if (error) ({ data, error } = await query());
  if (error) {
    throw new Error(`getFilerHoldings failed for ${filerName}: ${error.message}`);
  }

  return (data ?? []).map((r) => ({
    docId: r.doc_id,
    issuerCode: r.issuer_code,
    issuerName: r.issuer_name ?? r.issuer_code,
    holdingRatio: r.holding_ratio,
    holdingRatioPrior: r.holding_ratio_prior,
    discDate: r.disc_date,
    docTypeCode: r.doc_type_code,
    docDescription: r.doc_description,
  }));
}

export const getFilerHoldings = unstable_cache(getFilerHoldingsUncached, ["getFilerHoldings"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

// /investors（一覧）とサイトマップ用。edinet_filer_summary(Supabaseビュー)は
// filer_name単位で1行に集計済みだが、投資家は2,938件あり（2026-08-15時点）
// PostgRESTの既定1000行上限に掛かっていた。1000件しか返らないため一覧ページに出ないだけでなく、
// サイトマップからも約1,900件の投資家ページが丸ごと漏れていたため、ページングで全件取り切る。
// /investorsは`?category=`のsearchParamsを読むためリクエストごとの動的レンダリングになり
// ページ単位のrevalidateが効かない。実測でこのクエリだけで0.2〜1.0秒かかるため、
// 取得結果をunstable_cacheに載せて毎リクエストのSupabase往復を避ける。
export const getAllFilers = unstable_cache(
  async (): Promise<FilerSummary[]> => {
    const supabase = getSupabaseServerClient();
    const filers: FilerSummary[] = [];
    for (let page = 0; page < MAX_PAGES; page += 1) {
      const offset = page * PAGE_SIZE;
      const { data } = await supabase
        .from("edinet_filer_summary")
        .select("filer_name, category, holding_count, latest_disc_date")
        .order("latest_disc_date", { ascending: false })
        .order("filer_name", { ascending: true })
        .range(offset, offset + PAGE_SIZE - 1);
      if (!data || data.length === 0) break;
      filers.push(
        ...data.map((r) => ({
          filerName: r.filer_name,
          category: r.category as DealType,
          holdingCount: r.holding_count,
          latestDiscDate: r.latest_disc_date,
        }))
      );
      if (data.length < PAGE_SIZE) break;
    }
    return filers;
  },
  ["all-filers"],
  { revalidate: 3600 }
);

// /stocks/[code] からのクロスリンク用。この銘柄に大量保有報告書を提出したことがある
// 投資家一覧（名称+分類）を返す。
async function getFilersByStockCodeUncached(
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

export const getFilersByStockCode = unstable_cache(getFilersByStockCodeUncached, ["getFilersByStockCode"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

export type HoldingRow = {
  issuerCode: string;
  issuerName: string;
  filerName: string;
  discDate: string;
};

// 期間内の大量保有・変更報告書を全件返す（/trendingの期間比較と、提出者名の突合で共用）。
async function getHoldingsInRangeUncached(from: string, to: string): Promise<HoldingRow[]> {
  const supabase = getSupabaseServerClient();
  const rows: HoldingRow[] = [];
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const offset = page * PAGE_SIZE;
    const { data } = await supabase
      .from("edinet_large_holdings")
      .select("issuer_code, issuer_name, disc_date, filer_name")
      .gte("disc_date", from)
      .lte("disc_date", to)
      .order("disc_date", { ascending: true })
      .order("doc_id", { ascending: true })
      .range(offset, offset + PAGE_SIZE - 1);
    if (!data || data.length === 0) break;
    for (const row of data) {
      if (!row.issuer_code || !row.disc_date || !row.filer_name) continue;
      rows.push({
        issuerCode: row.issuer_code,
        issuerName: row.issuer_name ?? row.issuer_code,
        filerName: row.filer_name,
        discDate: row.disc_date,
      });
    }
    if (data.length < PAGE_SIZE) break;
  }
  return rows;
}

export const getHoldingsInRange = unstable_cache(getHoldingsInRangeUncached, ["getHoldingsInRange"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

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
  const rows = await getHoldingsInRange(from, to);

  const filersByKey = new Map<string, Set<string>>();
  for (const row of rows) {
    const key = `${row.issuerCode}|${row.discDate}`;
    const filers = filersByKey.get(key) ?? new Set<string>();
    filers.add(row.filerName);
    filersByKey.set(key, filers);
  }

  const resolved = new Map<string, string>();
  for (const [key, filers] of filersByKey) {
    if (filers.size === 1) resolved.set(key, [...filers][0]);
  }
  return resolved;
}

export type StockHoldingRow = {
  docId: string;
  filerName: string;
  discDate: string;
  holdingRatio: number | null;
  holdingRatioPrior: number | null;
  docTypeCode: string;
  docDescription: string | null;
};

// /stocks/[code]の「保有比率の推移」テーブル用。この銘柄に提出された大量保有・変更報告書を
// 開示日の新しい順に返す（提出者横断の時系列）。
async function getHoldingsByStockCodeUncached(stockCode: string): Promise<StockHoldingRow[]> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("edinet_large_holdings")
    .select("doc_id, filer_name, disc_date, holding_ratio, holding_ratio_prior, doc_type_code, doc_description")
    .eq("issuer_code", stockCode)
    .order("disc_date", { ascending: false })
    .limit(100);
  return (data ?? [])
    .filter((r) => r.filer_name)
    .map((r) => ({
      docId: r.doc_id,
      filerName: r.filer_name,
      discDate: r.disc_date,
      holdingRatio: r.holding_ratio,
      holdingRatioPrior: r.holding_ratio_prior,
      docTypeCode: r.doc_type_code,
      docDescription: r.doc_description,
    }));
}

export const getHoldingsByStockCode = unstable_cache(getHoldingsByStockCodeUncached, ["getHoldingsByStockCode"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

// 記事詳細のファクトボックス用。銘柄コード×開示日×提出者名でEDINET開示そのものを1件引き、
// 保有比率・直前の保有比率を返す（CMSの記事には保有比率フィールドが無いため）。
// 同一キーで複数行ある場合（同日の訂正等）は保有比率が取れている行を優先する。
async function getHoldingSnapshotUncached(
  stockCode: string,
  discDate: string,
  filerName: string
): Promise<{ holdingRatio: number | null; holdingRatioPrior: number | null } | null> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("edinet_large_holdings")
    .select("holding_ratio, holding_ratio_prior")
    .eq("issuer_code", stockCode)
    .eq("disc_date", discDate)
    .eq("filer_name", filerName)
    .order("holding_ratio", { ascending: false, nullsFirst: false })
    .limit(1)
    .maybeSingle();
  if (!data) return null;
  return { holdingRatio: data.holding_ratio, holdingRatioPrior: data.holding_ratio_prior };
}

export const getHoldingSnapshot = unstable_cache(getHoldingSnapshotUncached, ["getHoldingSnapshot"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

