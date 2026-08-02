import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import DealDateHeading from "@/components/DealDateHeading";
import Pagination from "@/components/Pagination";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { ARTICLES_PER_PAGE, getArticleList } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { CATEGORIES, DEAL_TYPE_BY_CATEGORY } from "@/types/article";

export function generateStaticParams() {
  return CATEGORIES.map((category) => ({ category }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ category: string }>;
}): Promise<Metadata> {
  const { category } = await params;
  const decodedCategory = decodeURIComponent(category);
  if (!DEAL_TYPE_BY_CATEGORY[decodedCategory]) return {};

  const title = `${decodedCategory}の記事一覧`;
  const description = `${decodedCategory}に関する大口取引の解説記事一覧。`;
  return {
    title,
    description,
    alternates: { canonical: `${SITE_URL}/category/${category}` },
    openGraph: { title, description, url: `${SITE_URL}/category/${category}` },
  };
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
  const groups = groupArticlesByDealDate(contents);

  return (
    <div>
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-gray-500">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-gray-700">{decodedCategory}</span>
      </nav>
      <h1 className="mb-6 text-2xl font-bold text-brand-navy">
        カテゴリ: {decodedCategory}
      </h1>
      {contents.length === 0 ? (
        <p className="text-gray-500">このカテゴリの記事がまだありません。</p>
      ) : (
        groups.map((group) => (
          <div key={group.date} className="mb-8">
            <DealDateHeading label={group.label} />
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              {group.articles.map((article) => (
                <ArticleCard key={article.id} article={article} />
              ))}
            </div>
          </div>
        ))
      )}
      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        basePath={`/category/${category}`}
      />
    </div>
  );
}
