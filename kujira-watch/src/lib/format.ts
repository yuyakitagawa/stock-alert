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
