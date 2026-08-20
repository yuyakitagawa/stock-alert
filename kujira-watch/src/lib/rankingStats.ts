import type { ArticleContent } from "@/types/article";

// /ranking/activist の集計。記事(microCMS)1件＝EDINET大量保有報告書1件の開示として扱う。
// 投資家別のランキング（/ranking/returns・/ranking/trending）はEDINET開示そのものを持つ
// Supabase側で集計するため、ここには記事ベースの銘柄別集計だけが残っている。

// 銘柄別ランキング（activist）の1行。
export type StockRow = {
  key: string;
  stockCode: string;
  stockName: string;
  amount: number;
  articleId: string;
  dealDate: string;
  filerName?: string;
  sell: boolean;
};

function isSell(article: ArticleContent): boolean {
  return (article.tags ?? "").split(",").some((tag) => tag.trim() === "売り");
}

export function buildStockRows(articles: ArticleContent[], limit: number): StockRow[] {
  return articles
    .filter((a) => a.dealType === "アクティビスト")
    .sort((x, y) => y.dealAmount - x.dealAmount)
    .slice(0, limit)
    .map((a) => ({
      key: a.id,
      stockCode: a.stockCode,
      stockName: a.stockName,
      amount: a.dealAmount,
      articleId: a.id,
      dealDate: a.dealDate,
      filerName: a.filerName,
      sell: isSell(a),
    }));
}
