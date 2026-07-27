import NewsRow from "@/components/NewsRow";
import Pagination from "@/components/Pagination";
import { NEWS_PER_PAGE, getNewsList } from "@/lib/microcms";

export default async function NewsListPage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string }>;
}) {
  const { page } = await searchParams;
  const currentPage = Math.max(1, Number(page) || 1);
  const offset = (currentPage - 1) * NEWS_PER_PAGE;

  const { contents, totalCount } = await getNewsList({ offset });
  const totalPages = Math.max(1, Math.ceil(totalCount / NEWS_PER_PAGE));

  return (
    <div className="mx-auto max-w-4xl px-4 py-16">
      <h1 className="mb-8 text-2xl font-bold text-brand-navy">ニュースリリース</h1>
      {contents.length === 0 ? (
        <p className="text-gray-500">お知らせはまだありません。</p>
      ) : (
        <div className="rounded-lg bg-white ring-1 ring-gray-200">
          {contents.map((news) => (
            <NewsRow key={news.id} news={news} />
          ))}
        </div>
      )}
      <Pagination currentPage={currentPage} totalPages={totalPages} basePath="/news" />
    </div>
  );
}
