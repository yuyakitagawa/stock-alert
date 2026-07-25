import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import Pagination from "@/components/Pagination";
import { ARTICLES_PER_PAGE, getArticleList } from "@/lib/microcms";
import { CATEGORIES, type Category } from "@/types/article";

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

  if (!CATEGORIES.includes(decodedCategory as Category)) {
    notFound();
  }

  const { page } = await searchParams;
  const currentPage = Math.max(1, Number(page) || 1);
  const offset = (currentPage - 1) * ARTICLES_PER_PAGE;

  const { contents, totalCount } = await getArticleList({
    offset,
    category: decodedCategory,
  });
  const totalPages = Math.max(1, Math.ceil(totalCount / ARTICLES_PER_PAGE));

  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold text-gray-900">
        カテゴリ: {decodedCategory}
      </h1>
      {contents.length === 0 ? (
        <p className="text-gray-500">このカテゴリの記事がまだありません。</p>
      ) : (
        <div className="flex flex-col gap-4">
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
