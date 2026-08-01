import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import Pagination from "@/components/Pagination";
import { ARTICLES_PER_PAGE, getArticleList } from "@/lib/microcms";
import { CATEGORIES, DEAL_TYPE_BY_CATEGORY } from "@/types/article";

export function generateStaticParams() {
  return CATEGORIES.map((category) => ({ category }));
}

export default async function CategoryPage({
  params,
  searchParams,
}: {
  params: Promise<{ category: string }>;
  searchParams: Promise<{ page?: string }>;
}) {
  const { category } = await params;
  const decodedCategory = decodeURIComponent(category);
  const dealType = DEAL_TYPE_BY_CATEGORY[decodedCategory];

  if (!dealType) {
    notFound();
  }

  const { page } = await searchParams;
  const currentPage = Math.max(1, Number(page) || 1);
  const offset = (currentPage - 1) * ARTICLES_PER_PAGE;

  const { contents, totalCount } = await getArticleList({
    offset,
    dealType,
  });
  const totalPages = Math.max(1, Math.ceil(totalCount / ARTICLES_PER_PAGE));

  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold text-brand-navy">
        カテゴリ: {decodedCategory}
      </h1>
      {contents.length === 0 ? (
        <p className="text-gray-500">このカテゴリの記事がまだありません。</p>
      ) : (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          {contents.map((article) => (
            <ArticleCard key={article.id} article={article} />
          ))}
        </div>
      )}
      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        basePath={`/category/${category}`}
      />
    </div>
  );
}
