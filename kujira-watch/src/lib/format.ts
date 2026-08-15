import type { Locale } from "@/lib/i18n";

export function formatDate(dateString: string, locale: Locale = "ja"): string {
  const date = new Date(dateString);
  if (locale === "en") {
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "2-digit",
    }).format(date);
  }
  return new Intl.DateTimeFormat("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

// "YYYY-MM" を「2026年8月」形式にする（/monthly の見出し・パンくず用）。
// Dateを介すとタイムゾーンで前月にずれる余地があるため、文字列のまま組み立てる。
export function formatMonth(month: string, locale: Locale = "ja"): string {
  const [year, mon] = month.split("-");
  if (locale === "en") {
    const label = new Intl.DateTimeFormat("en-US", { year: "numeric", month: "long", timeZone: "UTC" })
      .format(new Date(`${month}-01T00:00:00Z`));
    return label;
  }
  return `${year}年${Number(mon)}月`;
}

// amount は億円(100,000,000円)単位。英語版は短縮表記(¥X.XB / ¥XXXM)に変換する。
export function formatDealAmount(amount: number, locale: Locale = "ja"): string {
  if (locale === "en") {
    const millionYen = amount * 100;
    return millionYen >= 1000
      ? `¥${(millionYen / 1000).toFixed(1)}B`
      : `¥${millionYen.toFixed(0)}M`;
  }
  return `${amount.toLocaleString("ja-JP")}億円`;
}

// 億円単位の金額を表示用に「数字」と「単位」へ分ける。1兆円(1万億円)以上は兆円へ繰り上げ、
// 「+107,900.0億円」のような桁読みできない表記を避ける（/ranking・/weeklyの集計値用。
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
export function displayFilerName(name: string): string {
  return name
    .replace(/[０-９Ａ-Ｚａ-ｚ]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xfee0))
    .replace(/　/g, " ")
    .replace(/．/g, ".")
    .replace(/，/g, ",")
    .replace(/＆/g, "&")
    .replace(/ {2,}/g, " ")
    .trim();
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
export function linkifyFilerNames(html: string, filerNames: string[]): string {
  let result = html;
  for (const name of [...filerNames].sort((a, b) => b.length - a.length)) {
    if (!name) continue;
    for (const candidate of filerNameCandidates(name)) {
      const normalizedCandidate = candidate.normalize("NFKC");
      const normalizedResult = result.normalize("NFKC");
      const idx = normalizedResult.indexOf(normalizedCandidate);
      if (idx === -1) continue;
      const matchedText = result.slice(idx, idx + normalizedCandidate.length);
      const link = `<a href="/investors/${encodeURIComponent(name)}" class="text-brand-blue hover:underline">${matchedText}</a>`;
      result = result.slice(0, idx) + link + result.slice(idx + matchedText.length);
      break;
    }
  }
  return result;
}
