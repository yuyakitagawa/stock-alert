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
  // /investors/<番号> のURL用ID（edinet_filer_ids）。採番前の提出者はnull。
  filerId: number | null;
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
        .select("filer_name, category, holding_count, latest_disc_date, filer_id")
        .order("latest_disc_date", { ascending: false })
        .order("filer_name", { ascending: true })
        .range(offset, offset + PAGE_SIZE - 1);
      if (!data || data.length === 0) break;
      filers.push(
        ...data.map((r) => ({
          filerName: r.filer_name,
          filerId: r.filer_id ?? null,
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
// 投資家一覧（名称+分類+最新開示の保有比率）を返す。保有比率はその投資家の最新開示行の
// holding_ratio（訂正報告書で欠損していれば null）で、比率の高い順→名称順に並べる。
export type StockFiler = {
  filerName: string;
  filerId: number | null;
  category: DealType;
  latestRatio: number | null;
  latestDiscDate: string | null;
};

async function getFilersByStockCodeUncached(stockCode: string): Promise<StockFiler[]> {
  const supabase = getSupabaseServerClient();
  const { data: holdings } = await supabase
    .from("edinet_large_holdings")
    .select("filer_name, holding_ratio, disc_date")
    .eq("issuer_code", stockCode)
    .order("disc_date", { ascending: false });

  // 開示日降順で最初に現れた行＝最新開示。
  const latestByFiler = new Map<string, { ratio: number | null; discDate: string }>();
  for (const h of holdings ?? []) {
    if (!h.filer_name || latestByFiler.has(h.filer_name)) continue;
    latestByFiler.set(h.filer_name, { ratio: h.holding_ratio, discDate: h.disc_date });
  }
  const filerNames = [...latestByFiler.keys()];
  if (filerNames.length === 0) return [];

  const { data: classifications } = await supabase
    .from("edinet_filer_classification")
    .select("filer_name, category")
    .in("filer_name", filerNames);

  const categoryByFiler = new Map(
    (classifications ?? []).map((c) => [c.filer_name, c.category as DealType])
  );
  const idByFiler = await getFilerIdMap();
  return filerNames
    .map((filerName) => ({
      filerName,
      filerId: idByFiler[filerName] ?? null,
      category: categoryByFiler.get(filerName) ?? ("その他" as DealType),
      latestRatio: latestByFiler.get(filerName)?.ratio ?? null,
      latestDiscDate: latestByFiler.get(filerName)?.discDate ?? null,
    }))
    .sort(
      (a, b) =>
        (b.latestRatio ?? -1) - (a.latestRatio ?? -1) ||
        a.filerName.localeCompare(b.filerName, "ja")
    );
}

export const getFilersByStockCode = unstable_cache(getFilersByStockCodeUncached, ["getFilersByStockCode"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

// 開示1件の売買方向。「flat」は方向が判定できない開示（保有比率が動かない訂正報告書など）。
export type HoldingDirection = "buy" | "sell" | "flat";

// 保有比率が取れないときに概要欄から売り方向を拾うための語。
// トレーディングシステム側 tools/scan_large_holdings.py の _SELL_KEYWORDS と揃えてある。
const SELL_KEYWORDS = ["譲渡", "売却", "売出", "処分"];

// 保有比率とその前回値が両方取れるならその増減で判定し、取れない（初回の大量保有報告書など）
// ときだけ概要欄の語にフォールバックする。EDINETの概要欄は「変更報告書」とだけ書かれていて
// 売買方向を示さないことが多く、テキストだけに頼ると売り抜けを買いと誤判定するため
// （tools/scan_large_holdings.py: is_sell_disclosure() と同じ規律）。
export function holdingDirection(
  holdingRatio: number | null,
  holdingRatioPrior: number | null,
  docDescription: string | null
): HoldingDirection {
  if (holdingRatio !== null && holdingRatioPrior !== null) {
    if (holdingRatio > holdingRatioPrior) return "buy";
    if (holdingRatio < holdingRatioPrior) return "sell";
    return "flat";
  }
  const description = docDescription ?? "";
  return SELL_KEYWORDS.some((keyword) => description.includes(keyword)) ? "sell" : "buy";
}

export type HoldingRow = {
  // 推定売買金額(edinet_holding_amounts)を開示1件単位で突き合わせるためのキー。
  docId: string;
  issuerCode: string;
  issuerName: string;
  filerName: string;
  discDate: string;
  direction: HoldingDirection;
};

// 期間内の大量保有・変更報告書を全件返す（/trendingの期間比較と、提出者名の突合で共用）。
async function getHoldingsInRangeUncached(from: string, to: string): Promise<HoldingRow[]> {
  const supabase = getSupabaseServerClient();
  const rows: HoldingRow[] = [];
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const offset = page * PAGE_SIZE;
    const { data } = await supabase
      .from("edinet_large_holdings")
      .select("doc_id, issuer_code, issuer_name, disc_date, filer_name, holding_ratio, holding_ratio_prior, doc_description")
      .gte("disc_date", from)
      .lte("disc_date", to)
      .order("disc_date", { ascending: true })
      .order("doc_id", { ascending: true })
      .range(offset, offset + PAGE_SIZE - 1);
    if (!data || data.length === 0) break;
    for (const row of data) {
      if (!row.issuer_code || !row.disc_date || !row.filer_name) continue;
      rows.push({
        docId: row.doc_id,
        issuerCode: row.issuer_code,
        issuerName: row.issuer_name ?? row.issuer_code,
        filerName: row.filer_name,
        discDate: row.disc_date,
        direction: holdingDirection(row.holding_ratio, row.holding_ratio_prior, row.doc_description),
      });
    }
    if (data.length < PAGE_SIZE) break;
  }
  return rows;
}

// キーに版番号を足しているのは、新しいフィールドを持たない旧いキャッシュを引き継がないため
// （Vercelのデータキャッシュはデプロイをまたいで残るので、キーを変えないと最大1時間ぶん
// 空振りする）。v2=売買方向(direction)の追加、v3=doc_idの追加（推定売買金額の突き合わせ用）。
export const getHoldingsInRange = unstable_cache(getHoldingsInRangeUncached, ["getHoldingsInRange", "v3"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

// 開示1件ごとの推定売買金額（億円）。doc_idをキーに返す。
// EDINET開示そのものには取引金額が無く、「保有比率の変化幅 × 発行済株式数 × 開示日終値」で
// 概算する必要があるため、集計はSupabaseのマテリアライズドビュー側で完結させている
// （定義とレビュー観点は supabase/create_edinet_holding_amounts.sql）。
// 訂正報告書や、株価・発行済株式数が取れない開示はビューに行が無い＝金額不明で、
// 呼び出し側では0億円として扱う（件数には入るが金額には入らない）。
async function getHoldingAmountsInRangeUncached(from: string, to: string): Promise<Record<string, number>> {
  const supabase = getSupabaseServerClient();
  const amounts: Record<string, number> = {};
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const offset = page * PAGE_SIZE;
    const { data, error } = await supabase
      .from("edinet_holding_amounts")
      .select("doc_id, deal_amount_oku")
      .gte("disc_date", from)
      .lte("disc_date", to)
      .order("doc_id", { ascending: true })
      .range(offset, offset + PAGE_SIZE - 1);
    // 金額はランキングの並べ替え軸ではなく件数に添える情報なので、読み取りに失敗しても
    // ページ自体は件数だけで成立させる（0件を「金額ゼロ」として焼き付けない代わりに、
    // ここで投げてページを落とすことはしない）。
    if (error || !data || data.length === 0) break;
    for (const row of data) {
      if (row.deal_amount_oku === null) continue;
      amounts[row.doc_id] = Number(row.deal_amount_oku);
    }
    if (data.length < PAGE_SIZE) break;
  }
  return amounts;
}

// unstable_cacheはMapを返せない（キャッシュへの保存でJSON化されるため）ので
// プレーンオブジェクトで持ち回す。
export const getHoldingAmountsInRange = unstable_cache(
  getHoldingAmountsInRangeUncached,
  ["getHoldingAmountsInRange"],
  { revalidate: SUPABASE_REVALIDATE_SECONDS }
);

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

export type HoldingHistoryPoint = { discDate: string; holdingRatio: number };

// 記事詳細の「この投資家の保有比率の推移」用。同一銘柄×同一提出者の開示を古い順に返す。
// 1回ぶんの変化幅（前回比）だけでは「積み増している最中なのか、降りている最中なのか」が
// 分からないという指摘（2026-08-19のユーザーインタビュー）への対応。
async function getHoldingHistoryUncached(
  stockCode: string,
  filerName: string
): Promise<HoldingHistoryPoint[]> {
  const supabase = getSupabaseServerClient();
  const { data } = await supabase
    .from("edinet_large_holdings")
    .select("disc_date, holding_ratio")
    .eq("issuer_code", stockCode)
    .eq("filer_name", filerName)
    .order("disc_date", { ascending: true })
    .limit(50);
  const points: HoldingHistoryPoint[] = [];
  for (const row of data ?? []) {
    if (row.holding_ratio === null) continue;
    // 同一開示日に複数行（訂正等）がある場合は最後の値を採用する。
    const last = points[points.length - 1];
    if (last && last.discDate === row.disc_date) {
      last.holdingRatio = row.holding_ratio;
      continue;
    }
    points.push({ discDate: row.disc_date, holdingRatio: row.holding_ratio });
  }
  return points;
}

export const getHoldingHistory = unstable_cache(getHoldingHistoryUncached, ["getHoldingHistory"], {
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


// ---- 投資家ページのURL用ID ----
// 投資家ページのURLは /investors/<連番ID>（edinet_filer_ids）。以前は /investors/<提出者名>
// だったが、日本語・全角・空白を含む長いURLはGoogleにインデックスされず
// （2026-08-23のSearch Consoleカバレッジで投資家ページ603件が未登録）、事前生成時の
// ファイル名長制限(ENAMETOOLONG)にも掛かっていたため番号化した。
// 旧URL（提出者名）は /investors/[filer] 側で番号URLへ301転送する。

export { investorPath } from "@/lib/investorPath";

// 提出者名→IDの全件対応表。約3,000件で1回の読み取りに収まるため、リンクを大量に組み立てる
// ページ（銘柄ページの推移表・ランキング等）はこれを1度取って引く。
// unstable_cacheはJSON化して保存するため、MapではなくRecordで返す。
export const getFilerIdMap = unstable_cache(
  async (): Promise<Record<string, number>> => {
    const supabase = getSupabaseServerClient();
    const map: Record<string, number> = {};
    for (let page = 0; page < MAX_PAGES; page += 1) {
      const offset = page * PAGE_SIZE;
      const { data, error } = await supabase
        .from("edinet_filer_ids")
        .select("id, filer_name")
        .order("id", { ascending: true })
        .range(offset, offset + PAGE_SIZE - 1);
      if (error) throw new Error(`getFilerIdMap failed: ${error.message}`);
      if (!data || data.length === 0) break;
      for (const r of data) map[r.filer_name] = r.id;
      if (data.length < PAGE_SIZE) break;
    }
    return map;
  },
  ["filer-id-map"],
  { revalidate: SUPABASE_REVALIDATE_SECONDS }
);

async function getFilerNameByIdUncached(filerId: number): Promise<string | null> {
  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase
    .from("edinet_filer_ids")
    .select("filer_name")
    .eq("id", filerId)
    .maybeSingle();
  if (error) throw new Error(`getFilerNameById failed for ${filerId}: ${error.message}`);
  return data?.filer_name ?? null;
}

export const getFilerNameById = unstable_cache(getFilerNameByIdUncached, ["getFilerNameById"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});

async function getFilerIdByNameUncached(filerName: string): Promise<number | null> {
  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase
    .from("edinet_filer_ids")
    .select("id")
    .eq("filer_name", filerName)
    .maybeSingle();
  if (error) throw new Error(`getFilerIdByName failed for ${filerName}: ${error.message}`);
  return data?.id ?? null;
}

export const getFilerIdByName = unstable_cache(getFilerIdByNameUncached, ["getFilerIdByName"], {
  revalidate: SUPABASE_REVALIDATE_SECONDS,
});
