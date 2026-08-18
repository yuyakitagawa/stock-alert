import { unstable_cache } from "next/cache";
import { getSupabaseServerClient } from "@/lib/supabase";

// /activists（アクティビストの動き）用。投資家分類が「アクティビスト」の提出者について、
// 銘柄ごとの最新開示を「現在の保有」とみなして横断集計する。競合（アクティビストメディア等）の
// 「アクティビスト保有一覧」に対応するページの裏側。

// 最新開示の保有比率がこの値以上の銘柄を「保有中」とみなす。5%未満は大量保有報告の
// 義務を外れた（=手仕舞いまたは縮小）状態で、以降の実際の保有は開示されないため一覧に載せない。
export const ACTIVIST_HOLDING_MIN_RATIO = 5;

export type ActivistHolding = {
  filerName: string;
  issuerCode: string;
  issuerName: string;
  holdingRatio: number;
  discDate: string;
};

export type ActivistFund = {
  filerName: string;
  holdings: ActivistHolding[];
};

export type ActivistTargetStock = {
  issuerCode: string;
  issuerName: string;
  holders: ActivistHolding[];
};

export type ActivistHoldingsSummary = {
  funds: ActivistFund[];
  multiHolderStocks: ActivistTargetStock[];
  stockCount: number;
  holdingCount: number;
};

// PostgRESTの.in()はGETクエリ文字列に展開されるため、全角の長い提出者名51件を
// 一度に渡すとURLが長くなりすぎる。提出者名をこの件数ずつに分けて問い合わせる。
const FILER_CHUNK_SIZE = 15;
// 1チャンクあたりの行数はPostgREST既定の1000行上限に収まる想定だが、
// 念のためページングで取り切る（アクティビスト全体で2,076行、2026-08-16時点）。
const PAGE_SIZE = 1000;
const MAX_PAGES = 5;

export const getActivistHoldingsSummary = unstable_cache(
  async (): Promise<ActivistHoldingsSummary> => {
    const supabase = getSupabaseServerClient();

    const { data: activists } = await supabase
      .from("edinet_filer_classification")
      .select("filer_name")
      .eq("category", "アクティビスト");
    const filerNames = (activists ?? []).map((r) => r.filer_name).filter(Boolean);

    type Row = {
      filer_name: string;
      issuer_code: string | null;
      issuer_name: string | null;
      holding_ratio: number | null;
      disc_date: string | null;
      submit_date: string | null;
    };
    const rows: Row[] = [];
    for (let i = 0; i < filerNames.length; i += FILER_CHUNK_SIZE) {
      const chunk = filerNames.slice(i, i + FILER_CHUNK_SIZE);
      for (let page = 0; page < MAX_PAGES; page += 1) {
        const offset = page * PAGE_SIZE;
        const { data } = await supabase
          .from("edinet_large_holdings")
          .select("filer_name, issuer_code, issuer_name, holding_ratio, disc_date, submit_date")
          .in("filer_name", chunk)
          .order("disc_date", { ascending: false })
          .order("submit_date", { ascending: false })
          .order("doc_id", { ascending: false })
          .range(offset, offset + PAGE_SIZE - 1);
        if (!data || data.length === 0) break;
        rows.push(...data);
        if (data.length < PAGE_SIZE) break;
      }
    }

    // disc_date降順で並んでいるため、(提出者×銘柄)ごとに最初の行が最新開示。
    const latestByKey = new Map<string, ActivistHolding>();
    for (const row of rows) {
      if (!row.issuer_code || !row.disc_date || row.holding_ratio === null) continue;
      const key = `${row.filer_name}|${row.issuer_code}`;
      if (latestByKey.has(key)) continue;
      latestByKey.set(key, {
        filerName: row.filer_name,
        issuerCode: row.issuer_code,
        issuerName: row.issuer_name ?? row.issuer_code,
        holdingRatio: row.holding_ratio,
        discDate: row.disc_date,
      });
    }

    const current = [...latestByKey.values()].filter(
      (h) => h.holdingRatio >= ACTIVIST_HOLDING_MIN_RATIO
    );

    const fundMap = new Map<string, ActivistHolding[]>();
    const stockMap = new Map<string, ActivistHolding[]>();
    for (const holding of current) {
      fundMap.set(holding.filerName, [...(fundMap.get(holding.filerName) ?? []), holding]);
      stockMap.set(holding.issuerCode, [...(stockMap.get(holding.issuerCode) ?? []), holding]);
    }

    const funds: ActivistFund[] = [...fundMap.entries()]
      .map(([filerName, holdings]) => ({
        filerName,
        holdings: holdings.sort((a, b) => b.holdingRatio - a.holdingRatio),
      }))
      .sort(
        (a, b) =>
          b.holdings.length - a.holdings.length ||
          a.filerName.localeCompare(b.filerName, "ja")
      );

    const multiHolderStocks: ActivistTargetStock[] = [...stockMap.values()]
      .filter((holders) => holders.length >= 2)
      .map((holders) => ({
        issuerCode: holders[0].issuerCode,
        issuerName: holders[0].issuerName,
        holders: holders.sort((a, b) => b.holdingRatio - a.holdingRatio),
      }))
      .sort(
        (a, b) =>
          b.holders.length - a.holders.length ||
          b.holders[0].holdingRatio - a.holders[0].holdingRatio
      );

    return {
      funds,
      multiHolderStocks,
      stockCount: stockMap.size,
      holdingCount: current.length,
    };
  },
  ["activist-holdings-summary"],
  { revalidate: 3600 }
);

export type ActivistMove = {
  docId: string;
  filerName: string;
  issuerCode: string;
  issuerName: string;
  holdingRatio: number | null;
  holdingRatioPrior: number | null;
  discDate: string;
};

// /activists「直近の動き」用。直近days日にアクティビストが提出した開示を新しい順に返す。
// アクティビストの30日分は数十件規模なのでチャンクごとのページングは不要。
export const getActivistRecentMoves = unstable_cache(
  async (days: number): Promise<ActivistMove[]> => {
    const supabase = getSupabaseServerClient();

    const { data: activists } = await supabase
      .from("edinet_filer_classification")
      .select("filer_name")
      .eq("category", "アクティビスト");
    const filerNames = (activists ?? []).map((r) => r.filer_name).filter(Boolean);

    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - (days - 1));
    const cutoffDate = cutoff.toISOString().slice(0, 10);

    const moves: ActivistMove[] = [];
    for (let i = 0; i < filerNames.length; i += FILER_CHUNK_SIZE) {
      const chunk = filerNames.slice(i, i + FILER_CHUNK_SIZE);
      const { data } = await supabase
        .from("edinet_large_holdings")
        .select("doc_id, filer_name, issuer_code, issuer_name, holding_ratio, holding_ratio_prior, disc_date")
        .in("filer_name", chunk)
        .gte("disc_date", cutoffDate)
        .order("disc_date", { ascending: false })
        .order("doc_id", { ascending: false });
      for (const row of data ?? []) {
        if (!row.issuer_code || !row.disc_date) continue;
        moves.push({
          docId: row.doc_id,
          filerName: row.filer_name,
          issuerCode: row.issuer_code,
          issuerName: row.issuer_name ?? row.issuer_code,
          holdingRatio: row.holding_ratio,
          holdingRatioPrior: row.holding_ratio_prior,
          discDate: row.disc_date,
        });
      }
    }
    return moves.sort((a, b) =>
      a.discDate !== b.discDate
        ? b.discDate.localeCompare(a.discDate)
        : b.docId.localeCompare(a.docId)
    );
  },
  ["activist-recent-moves"],
  { revalidate: 3600 }
);
