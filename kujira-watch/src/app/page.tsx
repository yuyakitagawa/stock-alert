import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";

export default async function HomePage() {
  const { contents, totalCount } = await getArticleList();

  // 初回表示分（最新10件）のみをItemListとして構造化データ化する。オートスクロールで
  // 追加取得される分はクライアント側描画のためJSON-LDには含めない（クロール時点で
  // サーバーが返せる範囲と一致させる）。
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
        <InfiniteArticleList
          initialArticles={contents}
          totalCount={totalCount}
          showFeatured
        />
      )}
    </div>
  );
}
