import { investorPath } from "@/lib/investorPath";

export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

// time要素のdatetime属性用にYYYY-MM-DDへ揃える。microCMS/Supabase由来の日付は
// "2026-08-29" と "2026-08-29T00:00:00.000Z" が混在しており、後者をそのまま属性へ出すと
// 表示している日付（JST）と1日ずれることがある。
export function toDateAttr(dateString: string): string {
  return dateString.slice(0, 10);
}

// 一覧・集計ページが「いつまでのデータを反映しているか」を求める。ページの更新日には
// ビルド時刻ではなくこの値を使う（毎日ビルドしても中身が変わらない日はあるため）。
export function latestDateOf(dates: (string | null | undefined)[]): string | null {
  let latest: string | null = null;
  for (const d of dates) {
    if (!d) continue;
    const day = toDateAttr(d);
    if (!latest || day > latest) latest = day;
  }
  return latest;
}

// "YYYY-MM" を「2026年8月」形式にする（/monthly の見出し・パンくず用）。
// Dateを介すとタイムゾーンで前月にずれる余地があるため、文字列のまま組み立てる。
export function formatMonth(month: string): string {
  const [year, mon] = month.split("-");
  return `${year}年${Number(mon)}月`;
}

// amount は億円(100,000,000円)単位。
export function formatDealAmount(amount: number): string {
  return `${amount.toLocaleString("ja-JP")}億円`;
}

// 億円単位の金額を表示用に「数字」と「単位」へ分ける。1兆円(1万億円)以上は兆円へ繰り上げ、
// 「+107,900.0億円」のような桁読みできない表記を避ける（/weeklyの集計値用。
// 記事単体のdealAmountは兆に届かないためformatDealAmountのまま）。
export function formatAmountParts(
  amountOku: number,
  fractionDigits = 0
): { value: string; unit: "億円" | "兆円" } {
  if (Math.abs(amountOku) >= 10000) {
    return {
      value: (amountOku / 10000).toLocaleString("ja-JP", {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }),
      unit: "兆円",
    };
  }
  return {
    value: amountOku.toLocaleString("ja-JP", {
      minimumFractionDigits: fractionDigits,
      maximumFractionDigits: fractionDigits,
    }),
    unit: "億円",
  };
}

// EDINETの提出者名は英数字が全角（例:「ＢＣＰＥ　Ｐａｎｇｅａ　Ｃａｙｍａｎ，　Ｌ．Ｐ．」）で
// 登録されており、そのままでは読みにくい。表示専用に全角英数字と英文文脈の記号だけ半角へ寄せる。
// リンクhref・DB照合・JSON-LD・metadataは原文のまま使うこと（照合が壊れるため）。
// 日本語の句読点・中黒・全角括弧は変換しない。
export function displayText(text: string): string {
  return text
    .replace(/[０-９Ａ-Ｚａ-ｚ]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xfee0))
    .replace(/　/g, " ")
    .replace(/．/g, ".")
    .replace(/，/g, ",")
    .replace(/＆/g, "&")
    .replace(/ {2,}/g, " ")
    .trim();
}

export function displayFilerName(name: string): string {
  return displayText(name);
}

// 「※推測:」で始まる段落は事実ではなく解釈（web/publish_blog_articles.pyのプロンプトが
// 本文末尾に1文だけ書かせている）。地の文のまま流すと事実と混同されるため、
// 見出し付きの枠に切り出して「ここから先は推測」と分かる形にする。
const SPECULATION_RE = /<p>\s*※\s*推測\s*[:：]\s*([\s\S]*?)<\/p>/;

export function frameSpeculation(html: string): string {
  return html.replace(
    SPECULATION_RE,
    (_match, inner: string) =>
      '<aside class="not-prose my-6 rounded-md border border-rule border-l-4 border-l-brand-blue bg-section-tint px-4 py-3">' +
      '<p class="m-0 text-xs font-bold text-brand-navy">編集部の見立て（開示から読み取れる範囲の推測）</p>' +
      `<p class="m-0 mt-1 text-sm leading-relaxed text-ink-secondary">${inner}</p>` +
      "</aside>"
  );
}

export function excerptFromHtml(html: string, maxLength = 120): string {
  const text = html
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text;
}

// 売り方向（譲渡/売却）の記事はweb/publish_blog_articles.pyがtagsに"売り"を追加して
// 買いと区別する（microCMSのスキーマ変更を避けるため、既存の自由記述tagsフィールドを流用）。
export function isSellArticle(tags?: string): boolean {
  return (tags ?? "").split(",").map((tag) => tag.trim()).includes("売り");
}

// 訂正報告書（既に届け出た保有比率の事後訂正）の記事かどうか。
export function isCorrectionArticle(tags?: string): boolean {
  return (tags ?? "").split(",").map((tag) => tag.trim()).includes("訂正");
}

// 訂正報告書の記事は売買を伴わないため推定金額を持たない（web/publish_blog_articles.pyが
// dealAmount=0で投稿する）。金額の代わりに「訂正」と表示し、0億円の取引があったように
// 見せない。集計値（週次・月次の合計）は0が加算されるだけなので影響しない。
export function formatDealAmountOrCorrection(article: { dealAmount: number; tags?: string }): string {
  if (isCorrectionArticle(article.tags)) return "訂正";
  return formatDealAmount(article.dealAmount);
}

// DBの正式名（例:「シンフォニー・フィナンシャル・パートナーズ（シンガポール）ピーティーイー・
// リミテッド」）は法人格・所在地の注記が括弧書きで付くことが多いが、AI生成本文は読みやすさの
// ため括弧より前の通称だけで書くことがある。完全一致が無ければこの通称でも試す。
function filerNameCandidates(name: string): string[] {
  const candidates = [name];
  const beforeParen = name.split(/[（(]/)[0].trim();
  if (beforeParen && beforeParen !== name && beforeParen.length >= 4) {
    candidates.push(beforeParen);
  }
  return candidates;
}

// 記事本文（生成HTML）中の投資家名（初出のみ）を/investors/[filer]へのリンクに変換する。
// filerNameは自由記述本文の一部として埋め込まれておりCMS側に構造化フィールドが無いため、
// レンダリング時にこの銘柄の提出実績がある投資家名(getFilersByStockCode)と文字列突合する。
// EDINETのXBRLは提出者名を全角（Ｏａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ…）で保持する一方、
// 記事本文は半角で書かれるため、NFKC正規化した文字列上で位置を探し、本文側の表記
// （半角・通称）はそのまま残しつつ、リンク先だけDB上の正式表記（全角・フルネーム）でエンコードする。
export function linkifyFilerNames(
  html: string,
  filers: { filerName: string; filerId: number | null }[]
): string {
  let result = html;
  const idByName = new Map(filers.map((f) => [f.filerName, f.filerId]));
  for (const name of [...idByName.keys()].sort((a, b) => b.length - a.length)) {
    if (!name) continue;
    for (const candidate of filerNameCandidates(name)) {
      const normalizedCandidate = candidate.normalize("NFKC");
      const normalizedResult = result.normalize("NFKC");
      const idx = normalizedResult.indexOf(normalizedCandidate);
      if (idx === -1) continue;
      const matchedText = result.slice(idx, idx + normalizedCandidate.length);
      const link = `<a href="${investorPath(idByName.get(name), name)}" class="text-brand-blue hover:underline">${matchedText}</a>`;
      result = result.slice(0, idx) + link + result.slice(idx + matchedText.length);
      break;
    }
  }
  return result;
}
