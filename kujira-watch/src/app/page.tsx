import ArticleCard from "@/components/ArticleCard";
import DealDateHeading from "@/components/DealDateHeading";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import Pagination from "@/components/Pagination";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { ARTICLES_PER_PAGE, getArticleList } from "@/lib/microcms";
import { SITE_NAME } from "@/lib/site";

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
  const groups = groupArticlesByDealDate(showFeatured ? rest : contents);

  return (
    <div>
      <div className="mb-8 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200 sm:p-8">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">
          🐋 {SITE_NAME}へようこそ
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-gray-700">
          {SITE_NAME}は、EDINETの大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
          アクティビストファンド・インサイダー・自社株買いといった「クジラ」（相場を動かすほどの
          資金力を持つ大口投資家の俗称）が、どの銘柄をいつ・どれくらいの規模で動かしたかを
          日次でまとめて解説するブログです。個人投資家では追いきれない大口投資家の動きを、
          取引日ごとに一覧できます。
        </p>
      </div>
      {contents.length === 0 ? (
        <p className="text-gray-500">記事がまだありません。</p>
      ) : (
        <>
          {showFeatured && <FeaturedArticleCard article={featured} />}
          {groups.map((group) => (
            <div key={group.date} className="mb-8">
              <DealDateHeading label={group.label} />
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                {group.articles.map((article) => (
                  <ArticleCard key={article.id} article={article} />
                ))}
              </div>
            </div>
          ))}
        </>
      )}
      <Pagination currentPage={currentPage} totalPages={totalPages} basePath="/" />
    </div>
  );
}
