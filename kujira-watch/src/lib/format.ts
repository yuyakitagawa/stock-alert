export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

export function formatDealAmount(amount: number): string {
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
