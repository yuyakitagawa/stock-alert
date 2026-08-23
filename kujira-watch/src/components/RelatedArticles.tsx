import ArticleCard from "./ArticleCard";
import type { ArticleContent } from "@/types/article";

// ランキング・一覧系タブは表とテキストだけで画面が単調になるため、TOPと同じ
// アイキャッチ付き記事カードを各タブの文脈（そのタブに関係する記事）で添える。
// 記事が1件も無いときはセクションごと出さない（空見出しを残さない）。
export default function RelatedArticles({
  title,
  lead,
  articles,
}: {
  title: string;
  lead?: string;
  articles: ArticleContent[];
}) {
  if (articles.length === 0) return null;
  return (
    <section className="mb-10">
      <h2 className={`text-xl font-bold text-brand-navy ${lead ? "mb-1" : "mb-3"}`}>{title}</h2>
      {lead && <p className="mb-3 text-xs text-foreground/60">{lead}</p>}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        {articles.map((article) => (
          <ArticleCard key={article.id} article={article} />
        ))}
      </div>
    </section>
  );
}
