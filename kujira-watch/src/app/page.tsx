import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME } from "@/lib/site";

export default async function HomePage() {
  const { contents, totalCount } = await getArticleList();

  return (
    <div>
      <h1 className="sr-only">{SITE_NAME}｜新着記事</h1>
      {contents.length === 0 ? (
        <p className="text-gray-500">記事がまだありません。</p>
      ) : (
        <InfiniteArticleList
          initialArticles={contents}
          totalCount={totalCount}
          showFeatured
        />
      )}
      <div className="mt-10 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200 sm:p-8">
        <h2 className="text-xl font-bold text-brand-navy sm:text-2xl">
          🐋 {SITE_NAME}へようこそ
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-gray-700">
          {SITE_NAME}は、EDINETの大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
          アクティビストファンド・インサイダー・自社株買いといった「クジラ」（相場を動かすほどの
          資金力を持つ大口投資家の俗称）が、どの銘柄をいつ・どれくらいの規模で動かしたかを
          日次でまとめて解説するブログです。個人投資家では追いきれない大口投資家の動きを、
          取引日ごとに一覧できます。
        </p>
      </div>
    </div>
  );
}
