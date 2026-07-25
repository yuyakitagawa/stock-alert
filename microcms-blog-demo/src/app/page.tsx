import ArticleCard from "@/components/ArticleCard";
import Pagination from "@/components/Pagination";
import { ARTICLES_PER_PAGE, getArticleList } from "@/lib/microcms";

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

  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold text-gray-900">新着記事</h1>
      {contents.length === 0 ? (
        <p className="text-gray-500">記事がまだありません。</p>
      ) : (
        <div className="flex flex-col gap-4">
          {contents.map((article) => (
            <ArticleCard key={article.id} article={article} />
          ))}
        </div>
      )}
      <Pagination currentPage={currentPage} totalPages={totalPages} basePath="/" />
    </div>
  );
}
