import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME } from "@/lib/site";

// クローラーが最初のHTML(SSR)だけで辿れるリンク数を増やすため、初回取得件数を
// ARTICLES_PER_PAGE(10件・オートスクロールの追加取得単位)より多めにする。
// オートスクロールはJSでのみ発火するため、初回SSR分の実リンクがクロール可能な記事数の下限になる。
const INITIAL_ARTICLES_COUNT = 30;

export default async function HomePage() {
  const { contents, totalCount } = await getArticleList({ limit: INITIAL_ARTICLES_COUNT });

  return (
    <div>
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
