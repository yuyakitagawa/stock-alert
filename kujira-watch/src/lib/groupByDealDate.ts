import type { ArticleContent } from "@/types/article";
import { formatDate } from "./format";
import type { Locale } from "./i18n";

export type DealDateGroup = {
  date: string;
  label: string;
  articles: ArticleContent[];
};

// 記事一覧を取引日(dealDate)ごとにグループ化して見出しを付ける。
// 呼び出し側は事前にdealDate降順（同日内はdealAmount降順）でソート済みの配列を渡すこと
// （microcms.tsのgetArticleList/getArticlesByStockCodeのordersで担保）。
export function groupArticlesByDealDate(articles: ArticleContent[], locale: Locale = "ja"): DealDateGroup[] {
  const groups: DealDateGroup[] = [];
  for (const article of articles) {
    const date = article.dealDate.slice(0, 10);
    const last = groups[groups.length - 1];
    if (last && last.date === date) {
      last.articles.push(article);
    } else {
      groups.push({ date, label: formatDate(article.dealDate, locale), articles: [article] });
    }
  }
  return groups;
}
