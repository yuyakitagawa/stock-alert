import ArticleCard from "@/components/ArticleCard";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import Pagination from "@/components/Pagination";
import { ARTICLES_PER_PAGE, getArticleList } from "@/lib/microcms";
import { SITE_DESCRIPTION } from "@/lib/site";

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string }>;
}) {
  const { page } = await searchParams;
  const currentPage = Math.max(1, Number(page) || 1);
  const offset = (currentPage - 1) * ARTICLES_PER_PAGE;

  const { contents, totalCount } = await getArticleList({ offset });
  const totalPages = Math.max(1, Math.ceil(totalCount / ARTICLES_PER_PAGE));

  const [featured, ...rest] = contents;
  const showFeatured = currentPage === 1 && featured;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-brand-navy">新着記事</h1>
        <p className="mt-1 text-sm text-gray-500">{SITE_DESCRIPTION}</p>
      </div>
      {contents.length === 0 ? (
        <p className="text-gray-500">記事がまだありません。</p>
      ) : (
        <>
          {showFeatured && <FeaturedArticleCard article={featured} />}
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {(showFeatured ? rest : contents).map((article) => (
              <ArticleCard key={article.id} article={article} />
            ))}
          </div>
        </>
      )}
      <Pagination currentPage={currentPage} totalPages={totalPages} basePath="/" />
    </div>
  );
}
