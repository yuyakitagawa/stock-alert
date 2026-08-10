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

// 記事本文（生成HTML）中の投資家名（初出のみ）を/investors/[filer]へのリンクに変換する。
// filerNameは自由記述本文の一部として埋め込まれておりCMS側に構造化フィールドが無いため、
// レンダリング時にこの銘柄の提出実績がある投資家名(getFilersByStockCode)と文字列突合する。
// EDINETのXBRLは提出者名を全角（Ｏａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ…）で保持する一方、
// 記事本文は半角で書かれるため、NFKC正規化した文字列上で位置を探し、本文側の表記
// （半角）はそのまま残しつつ、リンク先だけDB上の正式表記（全角）でエンコードする。
export function linkifyFilerNames(html: string, filerNames: string[]): string {
  let result = html;
  for (const name of [...filerNames].sort((a, b) => b.length - a.length)) {
    if (!name) continue;
    const normalizedName = name.normalize("NFKC");
    const normalizedResult = result.normalize("NFKC");
    const idx = normalizedResult.indexOf(normalizedName);
    if (idx === -1) continue;
    const matchedText = result.slice(idx, idx + normalizedName.length);
    const link = `<a href="/investors/${encodeURIComponent(name)}" class="text-brand-blue hover:underline">${matchedText}</a>`;
    result = result.slice(0, idx) + link + result.slice(idx + matchedText.length);
  }
  return result;
}
