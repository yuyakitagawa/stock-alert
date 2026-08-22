import CategoryFilterDetails from "@/components/CategoryFilterDetails";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import FollowCta from "@/components/FollowCta";
import InfiniteArticleList from "@/components/InfiniteArticleList";
import TodayWhaleSummary from "@/components/TodayWhaleSummary";
import { getArticleList, getArticlesByDealDate, getFeaturedArticles } from "@/lib/microcms";
import { formatDate, isSellArticle } from "@/lib/format";
import { areDisclosuresFixed } from "@/lib/jst";
import { SITE_NAME, SITE_URL } from "@/lib/site";

// クローラーが最初のHTML(SSR)だけで辿れるリンク数を増やすため、初回取得件数を
// ARTICLES_PER_PAGE(10件・オートスクロールの追加取得単位)より多めにする。
// オートスクロールはJSでのみ発火するため、初回SSR分の実リンクがクロール可能な記事数の下限になる。
const INITIAL_ARTICLES_COUNT = 30;

export default async function HomePage() {
  const { contents, totalCount } = await getArticleList({ limit: INITIAL_ARTICLES_COUNT });
  const featuredArticles = contents.length > 0 ? await getFeaturedArticles() : [];
  const featuredIds = new Set(featuredArticles.map((a) => a.id));

  // 最新の取引日ぶんは、初回取得(INITIAL_ARTICLES_COUNT件)で切れている可能性があるため
  // 件数・金額は日付指定で取り直す（開示が多い日は1日で30件を超える）。
  const latestDealDate = contents[0]?.dealDate;
  const { contents: latestDayArticles } = latestDealDate
    ? await getArticlesByDealDate(latestDealDate.slice(0, 10))
    : { contents: [] };
  const latestDaySell = latestDayArticles.filter((a) => isSellArticle(a.tags));
  const latestDayBuy = latestDayArticles.filter((a) => !isSellArticle(a.tags));
  const sumAmount = (list: typeof latestDayArticles) => list.reduce((sum, a) => sum + a.dealAmount, 0);

  // 初回表示分（INITIAL_ARTICLES_COUNT件）のみをItemListとして構造化データ化する。
  // オートスクロールで追加取得される分はクライアント側描画のためJSON-LDには含めない
  // （クロール時点でサーバーが返せる範囲と一致させる）。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${SITE_NAME}｜新着記事`,
    itemListElement: contents.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: article.title,
      url: `${SITE_URL}/articles/${article.id}`,
    })),
  };

  return (
    <div>
      {contents.length > 0 && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
        />
      )}
      {/* H1は日付だけにせず主要検索語（大量保有報告書／大口投資家）を含める。
          初見の訪問者向けに「何のサイトか」を1文で添える（検索流入のほとんどは記事ページから
          入ってTOPへ上がってくるため、TOP単体でも自己紹介できるようにする）。 */}
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">
        {latestDealDate
          ? `大量保有報告書で読む大口投資家の動き（${formatDate(latestDealDate)}の取引）`
          : "大量保有報告書で読む大口投資家の動き"}
      </h1>
      <p className="mb-4 text-sm leading-relaxed text-foreground/70">
        EDINETに提出された大量保有報告書を平日毎時チェックし、機関投資家・アクティビスト・創業家など
        大口投資家の売買を開示当日のうちに記事にしています。
      </p>
      {contents.length === 0 ? (
        <p className="text-foreground/50">記事がまだありません。</p>
      ) : (
        <>
          {latestDealDate && latestDayArticles.length > 0 && (
            <TodayWhaleSummary
              date={latestDealDate}
              count={latestDayArticles.length}
              buyCount={latestDayBuy.length}
              buyAmount={sumAmount(latestDayBuy)}
              sellCount={latestDaySell.length}
              sellAmount={sumAmount(latestDaySell)}
              disclosuresFixed={areDisclosuresFixed(latestDealDate)}
            />
          )}
          {featuredArticles.length > 0 && (
            <div className="mb-8 space-y-4">
              {featuredArticles.map((article, i) => (
                <FeaturedArticleCard key={article.id} article={article} rank={i + 1} />
              ))}
            </div>
          )}
          {/* 記事ページにしか無かったフォロー導線をTOPにも置く。サイトの主要コンバージョンは
              Xフォロー（再訪のきっかけ）で、TOPは注目枠を読み終えた直後が最も関心が高い位置。 */}
          <div className="mb-8">
            <FollowCta />
          </div>
          {/* 一覧は最新の開示日ぶんとは限らない（土日は数日前が最新）ため「今日」と言い切らない。 */}
          <h2 className="mb-4 text-xl font-bold text-brand-navy">新着の取引</h2>
          <CategoryFilterDetails />
          <InfiniteArticleList
            initialArticles={contents}
            totalCount={totalCount}
            excludeIds={featuredIds}
          />
        </>
      )}
    </div>
  );
}
