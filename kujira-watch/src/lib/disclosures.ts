// Supabase edinet_large_holdings（EDINETの大量保有報告書・変更報告書・訂正報告書）を
// 読むための共通ヘルパー。開示の種別ラベルと原文PDFリンクを持つ。
// かつては全件を日付降順に並べる /disclosures（開示速報）があったが、同じ開示を
// TOPの記事一覧と二重に見せているだけだったため2026-08-18に廃止した
// （原文PDFへのリンクは /stocks/[code]・/investors/[filer] の開示履歴表へ移設）。
// 2026-09-06に記事の足切りを10億円へ上げたのに伴い、日付単位の対象開示一覧を
// /date/[date] の中に置いた（独立URLを作らないので二重表示の問題は再発しない）。

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

// ---- 保有目的・取得資金（大量保有報告書XBRLの本表）----
// EDINETのXBRLには保有割合のほかに保有目的・取得資金の内訳・保有株数が入っており、
// lib/edinet.py の parse_holding_details() が edinet_large_holdings に保存している。
// 判定ロジックは lib/edinet.py の classify_purpose() / average_acquisition_price() と同一。
// 片方だけ直すと記事本文（Python生成）とサイト表示（ここ）がずれるので必ず両方直すこと。

export type HoldingPurpose =
  | "重要提案行為等"
  | "経営参加"
  | "政策保有"
  | "安定株主"
  | "純投資";

// 株主としての姿勢が強い順に判定する。「純投資及び状況に応じて重要提案行為なども行う」
// のような留保付きの記載も、経営へ関与しうる立場を宣言している点で純投資とは区別する。
const PURPOSE_RULES: [HoldingPurpose, RegExp][] = [
  ["重要提案行為等", /重要提案行為|重要な提案|株主提案|企業価値の向上を目的として/],
  ["経営参加", /経営参加|経営に参画|経営権|子会社化|完全子会社|経営支援|役員として/],
  ["政策保有", /政策投資|政策保有|取引関係|業務提携|資本提携|関係強化|取引の維持|グループ間取引/],
  ["安定株主", /安定株主|長期保有|長期安定|経営の安定|継続保有/],
  // 運用会社・証券会社の「投資一任」「投資信託」「ディーリング」等はどれも経営に関与しない
  // 保有なので純投資に寄せる（直近553件で判定不可を77→36件に減らした）。
  ["純投資", /純投資|投資収益|値上がり|運用を目的|商品在庫|信託契約|投資一任|投資信託|信託財産|ディーリング|トレーディング|一時保有|一時的保有|資産運用/],
];

export function classifyPurpose(purposeText: string | null): HoldingPurpose | null {
  if (!purposeText) return null;
  for (const [label, pattern] of PURPOSE_RULES) {
    if (pattern.test(purposeText)) return label;
  }
  return null;
}

// バッジの色。経営に関与する側ほど強い色にして、一覧で目が留まるようにする。
// DEAL_TYPE_COLORS と同じOKLCHパレットの文字グレード（L 0.46）。保有目的バッジは
// ドットと文字の両方に同じ色を使うため、明るいドットグレードではなく文字グレードを持つ。
// 色相の意味づけも分類側とそろえてある（赤=関与が強い / 暖色=経営寄り / 青緑=事業上の保有 /
// 青=安定保有）。純投資は中立なので色相を持たせない。
export const PURPOSE_COLORS: Record<HoldingPurpose, string> = {
  重要提案行為等: "#9b0f1b",
  経営参加: "#903a03",
  政策保有: "#056662",
  安定株主: "#0e599d",
  純投資: "var(--ink-tertiary)",
};

export const PURPOSE_DESCRIPTIONS: Record<HoldingPurpose, string> = {
  重要提案行為等: "経営陣の選解任や事業方針への関与を目的に含む保有。実際に提案するかは別として、その立場を報告書で宣言している。",
  経営参加: "経営支援・子会社化など、経営そのものに関わることを目的とする保有。",
  政策保有: "取引関係の維持・強化など事業上の目的による保有。売買を目的としていない。",
  安定株主: "創業家・役員・取引先などが長期保有を明言している保有。",
  純投資: "値上がり益や運用収益を目的とする保有。経営への関与は目的に含まない。",
};

// 開示ベースの平均取得単価（円/株）= 取得資金の総額 ÷ 保有株数。
// 「現在保有している分」の取得原価なので、政策保有株のように保有が古いほど
// 現在株価から乖離する（それ自体が含み損益の手がかりになる）。
export function averageAcquisitionPrice(
  fundingTotal: number | null,
  sharesHeld: number | null
): number | null {
  if (!fundingTotal || !sharesHeld || sharesHeld <= 0) return null;
  return Math.round((fundingTotal / sharesHeld) * 10) / 10;
}

// 取得資金に占める借入金の割合（%）。自己資金0＝全額借入の買いが実在し、
// 返済圧力がある分だけ同じ保有比率でも意味が違う。
export function borrowingRatio(
  fundingBorrowings: number | null,
  fundingTotal: number | null
): number | null {
  if (!fundingTotal || fundingTotal <= 0 || fundingBorrowings === null) return null;
  // 借入金が取得資金の総額を上回る開示が実在する（LEOMO,inc→日本製麻は借入6.30億円に対し
  // 取得資金の総額6.23億円。総額が処分ぶんを差し引いた残高で、借入は総額で書かれるため）。
  // 「101.2%」は読み手にはサイトのバグに見えるので、比率としては100%＝全額借入に丸める。
  return Math.min(100, Math.round((fundingBorrowings / fundingTotal) * 1000) / 10);
}

// 報告義務発生日（実際に5%を超えた日）から提出日までの日数。
// 法定は5営業日以内だが、実際には数ヶ月〜1年遅れる開示がある。
export function filingLagDays(obligationDate: string | null, discDate: string | null): number | null {
  if (!obligationDate || !discDate) return null;
  const lag = Math.round((Date.parse(discDate) - Date.parse(obligationDate)) / 86400000);
  return Number.isFinite(lag) && lag >= 0 ? lag : null;
}

// ---- 「この日の対象開示」の判定（/date/[date]）----
// web/market_timing_alert.py の get_recent_large_holdings() と同じ足切りをTS側に置いたもの。
// 記事の足切り（推定売買金額10億円以上）を通らなかった開示も /date/[date] に全件並べるため、
// 「記事にはしないが対象ではある開示」をここで選び直す必要がある。
// 数値・判定順は tools/scan_large_holdings.py の is_noise_match() / is_material_correction() と
// 同一にすること。片方だけ直すと「一覧に出ているのに記事化されない理由が説明できない」状態になる。
//
// Python側との唯一の差: 自己申告の判定に使う発行体名が、Pythonは data/code_name_map.json
// （J-Quantsの銘柄名）優先なのに対しここはEDINETのissuer_nameのみ。どちらも法人格を
// 除去して包含判定するため実務上の差は出ないが、完全一致ではない。

/** これ以上の保有比率はスクイーズアウト対象になりうる水準で、大口の「買い集め」として扱わない。 */
export const MAJORITY_HOLDING_THRESHOLD = 51;
/** 訂正報告書のうち、届出比率がこのポイント数以上動くものは既報の誤りを正す情報として扱う。 */
export const MATERIAL_CORRECTION_DELTA_PT = 3.0;
/** 記事にする推定売買金額の下限（億円）。web/publish_blog_articles.py の MIN_DEAL_AMOUNT_OKU と同値。 */
export const ARTICLE_MIN_DEAL_AMOUNT_OKU = 10;

const NAME_NOISE_TOKENS = ["株式会社", "(株)", "（株）", "ホールディングス", "HD", " ", "　", "・"];

function normalizeCompanyName(name: string): string {
  let out = name ?? "";
  for (const token of NAME_NOISE_TOKENS) out = out.split(token).join("");
  return out.trim();
}

function isCorrectionReport(docDescription: string | null): boolean {
  return (docDescription ?? "").includes("訂正");
}

// 概要欄が売買方向を示さない開示が多いため、直前保有割合が取れるときは比率の増減で判定する。
const SELL_KEYWORDS = ["譲渡", "売却", "売出", "処分"];

function isSellDisclosure(
  docDescription: string | null,
  ratio: number | null,
  prior: number | null
): boolean {
  if (ratio !== null && prior !== null) return ratio < prior;
  return SELL_KEYWORDS.some((k) => (docDescription ?? "").includes(k));
}

export type DisclosureFilterInput = {
  filerName: string;
  issuerName: string;
  docDescription: string | null;
  holdingRatio: number | null;
  holdingRatioPrior: number | null;
};

/** 訂正報告書のうち届出比率が大きく動くもの（既報の保有比率そのものが誤りだった開示）。 */
export function isMaterialCorrection(row: DisclosureFilterInput): boolean {
  if (!isCorrectionReport(row.docDescription)) return false;
  if (row.holdingRatio === null || row.holdingRatioPrior === null) return false;
  return Math.abs(row.holdingRatio - row.holdingRatioPrior) >= MATERIAL_CORRECTION_DELTA_PT;
}

/**
 * サイトが扱う対象開示か。除外するのは
 * 過半数超（51%以上）・訂正報告書（大幅訂正を除く）・自己申告（提出者≒発行体）の3つ。
 * 売り（保有比率の減少）は大口の動きとして扱うので除外しない。
 */
export function isTargetDisclosure(row: DisclosureFilterInput): boolean {
  if (row.holdingRatio !== null && Math.abs(row.holdingRatio) >= MAJORITY_HOLDING_THRESHOLD) {
    return false;
  }
  if (isCorrectionReport(row.docDescription)) return isMaterialCorrection(row);
  // Python側は売り判定が自己申告判定より先に来る。順序を変えると
  // 「自己申告かつ売り」の開示の扱いがずれるため、ここでも売りを先に通す。
  if (isSellDisclosure(row.docDescription, row.holdingRatio, row.holdingRatioPrior)) return true;
  const filer = normalizeCompanyName(row.filerName);
  const issuer = normalizeCompanyName(row.issuerName);
  if (filer && issuer && (filer.includes(issuer) || issuer.includes(filer))) return false;
  return true;
}
