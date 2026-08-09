import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList, getFeaturedArticles } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";

// クローラーが最初のHTML(SSR)だけで辿れるリンク数を増やすため、初回取得件数を
// ARTICLES_PER_PAGE(10件・オートスクロールの追加取得単位)より多めにする。
// オートスクロールはJSでのみ発火するため、初回SSR分の実リンクがクロール可能な記事数の下限になる。
const INITIAL_ARTICLES_COUNT = 30;

export default async function HomePage() {
  const { contents, totalCount } = await getArticleList({ limit: INITIAL_ARTICLES_COUNT });
  const featuredArticles = contents.length > 0 ? await getFeaturedArticles() : [];
  const featuredIds = new Set(featuredArticles.map((a) => a.id));

  // 初回表示分（INITIAL_ARTICLES_COUNT件）のみをItemListとして構造化データ化する。
  // オートスクロールで追加取得される分はクライアント側描画のためJSON-LDには含めない
  // （クロール時点でサーバーが返せる範囲と一致させる）。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${SITE_NAME}｜新着記事`,
    itemListElement: contents.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: article.title,
      url: `${SITE_URL}/articles/${article.id}`,
    })),
  };

  return (
    <div>
      {contents.length > 0 && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
        />
      )}
      <h1 className="sr-only">{SITE_NAME}｜新着記事</h1>
      {contents.length === 0 ? (
        <p className="text-foreground/50">記事がまだありません。</p>
      ) : (
        <>
          {featuredArticles.length > 0 && (
            <div className="mb-8 space-y-4">
              {featuredArticles.map((article, i) => (
                <FeaturedArticleCard key={article.id} article={article} rank={i + 1} />
              ))}
            </div>
          )}
          <InfiniteArticleList
            initialArticles={contents}
            totalCount={totalCount}
            excludeIds={featuredIds}
          />
        </>
      )}
    </div>
  );
}
