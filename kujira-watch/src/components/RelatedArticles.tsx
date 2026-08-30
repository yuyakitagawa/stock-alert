import ArticleCard from "./ArticleCard";
import type { ArticleContent } from "@/types/article";

// ランキング・一覧系タブは表とテキストだけで画面が単調になるため、TOPと同じ
// アイキャッチ付き記事カードを各タブの文脈（そのタブに関係する記事）で添える。
// 記事が1件も無いときはセクションごと出さない（空見出しを残さない）。
//
// 2026-08-29: 同サイズのカード4件を2列に並べるだけだったのを、先頭1件を2列ぶんの幅に
// 広げる形へ変更し、件数も8件までに増やした。データページは4ページとも
// 「見出し→細いグレーの説明文→小さいカードの羅列」で、画像が最下部の4枚しか無く
// 見た目が弱かったため。素材は既存のmicroCMSアイキャッチをそのまま使うので追加制作は無い。
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
      {lead && <p className="mb-3 text-xs text-ink-tertiary">{lead}</p>}
      {/* 3列グリッドの先頭だけ2列ぶんを占有させる。2件目以降は通常サイズのまま流し込む
          （8件なら 大+1 / 3 / 3 の並びになる）。スマホでは全件1列。 */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {articles.map((article, index) => (
          <div key={article.id} className={index === 0 ? "sm:col-span-2" : undefined}>
            <ArticleCard article={article} />
          </div>
        ))}
      </div>
    </section>
  );
}
