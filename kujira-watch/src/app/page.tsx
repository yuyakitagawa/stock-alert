import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME } from "@/lib/site";

export default async function HomePage() {
  const { contents, totalCount } = await getArticleList();

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
