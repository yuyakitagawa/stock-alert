import { unstable_cache } from "next/cache";
import { getSupabaseServerClient } from "@/lib/supabase";

// /disclosures（開示速報）用。記事化された開示だけでなく、EDINETに提出された
// 大量保有報告書・変更報告書の全件をSupabase edinet_large_holdingsから直接一覧する。
// 競合（IRBANK・M&A Online等）が持つ「全開示の速報一覧」に対応するページの裏側。

export const DISCLOSURES_PER_PAGE = 100;

// doc_type_codeは350=大量保有報告書・変更報告書の両方 / 360=訂正報告書で、
// 新規と変更はdoc_descriptionの接頭辞（「大量保有報告書…」「変更報告書…」）でしか
// 区別できない（2026-08-16にSQLで全19,799件の内訳を確認: 新規2,743 / 変更13,927 / 訂正3,129）。
export type DisclosureTypeFilter = "new" | "change" | "correction";

const DESCRIPTION_PATTERN_BY_FILTER: Record<Exclude<DisclosureTypeFilter, "correction">, string> = {
  new: "大量保有報告書%",
  change: "変更報告書%",
};

export type DisclosureRow = {
  docId: string;
  filerName: string;
  docDescription: string | null;
  submitDate: string;
  discDate: string;
  holdingRatio: number | null;
  holdingRatioPrior: number | null;
  issuerCode: string;
  issuerName: string;
  docTypeCode: string;
};

// EDINETの書類PDFへの直リンク（閲覧サイトの検索を経由せず原文を開ける）。
export function edinetPdfUrl(docId: string): string {
  return `https://disclosure2dl.edinet-fsa.go.jp/searchdocument/pdf/${docId}.pdf`;
}

// 行の種別ラベル。doc_descriptionの接頭辞で判定し、descriptionが無い行だけ
// doc_type_code（360=訂正）にフォールバックする。
export function disclosureKindLabel(row: {
  docDescription: string | null;
  docTypeCode: string;
}): "新規" | "変更" | "訂正" {
  if (row.docDescription?.startsWith("訂正")) return "訂正";
  if (row.docDescription?.startsWith("変更報告書")) return "変更";
  if (row.docDescription?.startsWith("大量保有報告書")) return "新規";
  return row.docTypeCode === "360" ? "訂正" : "新規";
}

// /investors/[filer]・/stocks/[code]の推移テーブル用の報告書名ラベル。
// 以前はdoc_type_codeだけで「350=大量保有報告書/360=変更報告書」と表示していたが、
// 実際は350が新規・変更の両方、360が訂正報告書で、変更報告書（全体の7割）を
// 「大量保有報告書」、訂正を「変更報告書」と誤表示していた。
const DOC_LABEL_BY_KIND = {
  新規: "大量保有報告書",
  変更: "変更報告書",
  訂正: "訂正報告書",
} as const;

export function disclosureDocLabel(row: {
  docDescription: string | null;
  docTypeCode: string;
}): string {
  return DOC_LABEL_BY_KIND[disclosureKindLabel(row)];
}

// /disclosuresはsearchParams（?page=/?type=）を読むためリクエストごとの動的レンダリングになり、
// ページ単位のrevalidateが効かない。unstable_cacheは引数もキーに含まれるため、
// (page, type)の組ごとに5分キャッシュしてSupabase往復を避ける。
export const getDisclosuresPage = unstable_cache(
  async (page: number, type: DisclosureTypeFilter | null): Promise<DisclosureRow[]> => {
    const supabase = getSupabaseServerClient();
    const offset = (page - 1) * DISCLOSURES_PER_PAGE;
    let query = supabase
      .from("edinet_large_holdings")
      .select(
        "doc_id, filer_name, doc_description, submit_date, disc_date, holding_ratio, holding_ratio_prior, issuer_code, issuer_name, doc_type_code"
      )
      .order("disc_date", { ascending: false })
      .order("submit_date", { ascending: false })
      .order("doc_id", { ascending: false })
      .range(offset, offset + DISCLOSURES_PER_PAGE - 1);
    if (type === "correction") query = query.eq("doc_type_code", "360");
    else if (type) query = query.like("doc_description", DESCRIPTION_PATTERN_BY_FILTER[type]);
    const { data } = await query;

    return (data ?? [])
      .filter((r) => r.doc_id && r.filer_name && r.issuer_code && r.disc_date)
      .map((r) => ({
        docId: r.doc_id,
        filerName: r.filer_name,
        docDescription: r.doc_description,
        submitDate: r.submit_date ?? r.disc_date,
        discDate: r.disc_date,
        holdingRatio: r.holding_ratio,
        holdingRatioPrior: r.holding_ratio_prior,
        issuerCode: r.issuer_code,
        issuerName: r.issuer_name ?? r.issuer_code,
        docTypeCode: r.doc_type_code,
      }));
  },
  ["disclosures-page"],
  { revalidate: 300 }
);

export type DisclosureCounts = {
  all: number;
  new: number;
  change: number;
  correction: number;
};

// フィルターボタンの件数表示とページ数計算用。head:trueで行本体は転送しない。
export const getDisclosureCounts = unstable_cache(
  async (): Promise<DisclosureCounts> => {
    const supabase = getSupabaseServerClient();
    const base = () =>
      supabase.from("edinet_large_holdings").select("doc_id", { count: "exact", head: true });
    const [allRes, newRes, changeRes, correctionRes] = await Promise.all([
      base(),
      base().like("doc_description", DESCRIPTION_PATTERN_BY_FILTER.new),
      base().like("doc_description", DESCRIPTION_PATTERN_BY_FILTER.change),
      base().eq("doc_type_code", "360"),
    ]);
    return {
      all: allRes.count ?? 0,
      new: newRes.count ?? 0,
      change: changeRes.count ?? 0,
      correction: correctionRes.count ?? 0,
    };
  },
  ["disclosures-counts"],
  { revalidate: 300 }
);
