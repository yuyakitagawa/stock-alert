import { unstable_cache } from "next/cache";
import { getSupabaseServerClient } from "@/lib/supabase";

// Supabase edinet_large_holdings（EDINETの大量保有報告書・変更報告書・訂正報告書）を
// 読むための共通ヘルパー。開示の種別ラベル・原文PDFリンクと、/trending・/weeklyの
// 件数トレンド集計をまとめている。
// かつては全件を日付降順に並べる /disclosures（開示速報）があったが、同じ開示を
// TOPの記事一覧と二重に見せているだけだったため2026-08-18に廃止した
// （原文PDFへのリンクは /stocks/[code]・/investors/[filer] の開示履歴表へ移設）。

// EDINETの書類PDFへの直リンク（閲覧サイトの検索を経由せず原文を開ける）。
export function edinetPdfUrl(docId: string): string {
  return `https://disclosure2dl.edinet-fsa.go.jp/searchdocument/pdf/${docId}.pdf`;
}

// 行の種別ラベル。doc_type_codeは350=大量保有報告書・変更報告書の両方 / 360=訂正報告書で、
// 新規と変更はdoc_descriptionの接頭辞（「大量保有報告書…」「変更報告書…」）でしか
// 区別できない（2026-08-16にSQLで全19,799件の内訳を確認: 新規2,743 / 変更13,927 / 訂正3,129）。
// descriptionが無い行だけdoc_type_code（360=訂正）にフォールバックする。
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

export type MonthlyDisclosureCount = {
  month: string; // "YYYY-MM"
  count: number;
  isPartial: boolean; // 当月（集計中）
};

// /trendingの「月別の開示件数トレンド」用。月ごとの開示件数をhead:trueのcountクエリで集計する
// （PostgRESTにGROUP BYが無いため月数ぶんの並列カウントで代替。月数は高々十数個）。
// データ取得開始月（2025-06、月の途中から）は件数が過少で傾向を誤読させるため除外し、
// 当月はisPartial=trueで「集計中」の表示に使う。
export const getMonthlyDisclosureCounts = unstable_cache(
  async (): Promise<MonthlyDisclosureCount[]> => {
    const supabase = getSupabaseServerClient();
    const { data: first } = await supabase
      .from("edinet_large_holdings")
      .select("disc_date")
      .order("disc_date", { ascending: true })
      .limit(1)
      .maybeSingle();
    if (!first?.disc_date) return [];

    const currentMonth = new Date().toISOString().slice(0, 7);
    const months: string[] = [];
    // データ開始月の翌月から当月まで。
    const cursor = new Date(`${first.disc_date.slice(0, 7)}-01T00:00:00Z`);
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
    while (cursor.toISOString().slice(0, 7) <= currentMonth) {
      months.push(cursor.toISOString().slice(0, 7));
      cursor.setUTCMonth(cursor.getUTCMonth() + 1);
    }

    const counts = await Promise.all(
      months.map(async (month) => {
        const next = new Date(`${month}-01T00:00:00Z`);
        next.setUTCMonth(next.getUTCMonth() + 1);
        const { count } = await supabase
          .from("edinet_large_holdings")
          .select("doc_id", { count: "exact", head: true })
          .gte("disc_date", `${month}-01`)
          .lt("disc_date", `${next.toISOString().slice(0, 7)}-01`);
        return { month, count: count ?? 0, isPartial: month === currentMonth };
      })
    );
    return counts;
  },
  ["monthly-disclosure-counts"],
  { revalidate: 3600 }
);
