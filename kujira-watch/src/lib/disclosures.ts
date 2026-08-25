// Supabase edinet_large_holdings（EDINETの大量保有報告書・変更報告書・訂正報告書）を
// 読むための共通ヘルパー。開示の種別ラベルと原文PDFリンクを持つ。
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

// ---- 短期大量譲渡（法第27条の25第2項）の「譲渡の相手方・単価」----
// EDINETは通常「誰にいくらで売ったか」を開示しないが、短期大量譲渡に該当する変更報告書だけは
// 直近60日間の取得・処分が相手方・単価つきの表で載る。Supabaseの
// edinet_large_holdings.short_term_transfers に取り込んであり（lib/edinet.py で解析）、
// このサイトのファクトボックスに出す。集計ロジックはlib/edinet.pyのsummarize_disposals()と同一。
export type ShortTermTransfer = {
  date: string | null;
  security_type: string | null;
  shares: number | null;
  ratio: number | null;
  venue: string | null;
  action: string | null;
  counterparty: string | null;
  unit_price: number | null;
  unit_price_note: string | null;
};

export type TransferSummary = {
  counterparties: string[];
  amountOku: number | null;
  shares: number;
  unitPrice: number | null;
  // 新株予約権・社債等は「1株いくら」ではないので、表示側で単位を出し分ける
  securityType: string | null;
  isEquity: boolean;
  date: string | null;
  venue: string | null;
  rows: ShortTermTransfer[];
};

// 「市場内取引のため不明」「該当なし」等、相手方が特定できない表記
function isUnknownCounterparty(name: string | null): boolean {
  return !name || name.includes("不明") || name.includes("該当なし");
}

// 単価が「株価」と言えるのは株券（株式）の行だけ。新株予約権証券・社債券等は1個あたりの価格。
const NON_EQUITY = /新株予約権|社債|預託証券|受益証券|カバードワラント|転換/;
const EQUITY = /株券|株式|普通株/;

function isEquityRow(row: ShortTermTransfer): boolean {
  const kind = row.security_type ?? "";
  return EQUITY.test(kind) && !NON_EQUITY.test(kind);
}

export function summarizeDisposals(
  transfers: ShortTermTransfer[] | null,
  ratioChange: number | null
): TransferSummary | null {
  if (!transfers || transfers.length === 0) return null;
  const disposals = transfers.filter((t) => t.action === "処分");
  const acquisitions = transfers.filter((t) => t.action === "取得");
  const priced = disposals.filter((t) => t.unit_price && t.shares);
  const named = disposals.filter((t) => !isUnknownCounterparty(t.counterparty));
  if (named.length === 0 && priced.length === 0) return null;

  // 実額に使えるのは株式（株券）の譲渡だけ。新株予約権・社債等は単価が株価ではない。
  const equity = priced.filter(isEquityRow);
  // 実額として使うのは「表の処分行だけで今回の比率変化を説明できる」ときだけ。
  // 60日間の売買が細かく並ぶ開示（取得と処分が混在）は差引きが表から復元できない。
  let used: ShortTermTransfer[] = [];
  if (equity.length > 0 && acquisitions.length === 0) {
    const ratioSum = equity.reduce((sum, t) => sum + (t.ratio ?? 0), 0);
    if (ratioChange === null || Math.abs(ratioSum - Math.abs(ratioChange)) <= 0.5) {
      used = equity;
    } else {
      // 表は直近60日ぶんなので、連続して売った提出者の2枚目以降には前回開示済みの行も並ぶ。
      // 今回の変化幅と一致する行が1つだけならその行が今回の譲渡。
      const match = equity.filter((t) => Math.abs((t.ratio ?? 0) - Math.abs(ratioChange)) <= 0.2);
      if (match.length === 1) used = match;
    }
  }
  const amountYen = used.reduce((sum, t) => sum + t.unit_price! * t.shares!, 0);
  // 0.05億円未満は四捨五入で「0.0億円」になり取引が無かったように見えるため採らない
  if (amountYen < 5e6) used = [];
  // 実額の内訳が確定したときはその相手方だけを見せる（表に載る過去の相手方を並べない）
  const shownSource = used.length > 0 ? used : named;
  const shown = shownSource.filter((t) => !isUnknownCounterparty(t.counterparty));
  const rows = shown.length > 0 ? shown : named;
  // 単価・日付・市場内外は最大株数の行（＝その開示の主役の取引）を代表値にする
  const pool = used.length > 0 ? used : priced.length > 0 ? priced : named;
  const main = pool.reduce((a, b) => ((a.shares ?? 0) >= (b.shares ?? 0) ? a : b), pool[0]);
  return {
    counterparties: Array.from(new Set(rows.map((t) => t.counterparty!))),
    amountOku: used.length > 0 ? Math.round((amountYen / 1e8) * 10) / 10 : null,
    shares: (used.length > 0 ? used : priced).reduce((sum, t) => sum + (t.shares ?? 0), 0),
    unitPrice: main?.unit_price ?? null,
    securityType: main?.security_type ?? null,
    isEquity: main ? isEquityRow(main) : false,
    date: main?.date ?? null,
    venue: main?.venue ?? null,
    rows,
  };
}
