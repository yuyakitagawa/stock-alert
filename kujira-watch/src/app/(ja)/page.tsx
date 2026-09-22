import CategoryFilterDetails from "@/components/CategoryFilterDetails";
import DataUpdatedAt from "@/components/DataUpdatedAt";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import InfiniteArticleList from "@/components/InfiniteArticleList";
import Link from "next/link";
import StockSearch from "@/components/StockSearch";
import TopTrendingPreview from "@/components/TopTrendingPreview";
import { getArticleList, getFeaturedArticle } from "@/lib/microcms";
import { getPublishedDates } from "@/lib/publishedPages";
import { SITE_NAME, SITE_URL } from "@/lib/site";

// クローラーが最初のHTML(SSR)だけで辿れるリンク数を増やすため、初回取得件数を
// ARTICLES_PER_PAGE(10件・オートスクロールの追加取得単位)より多めにする。
// オートスクロールはJSでのみ発火するため、初回SSR分の実リンクがクロール可能な記事数の下限になる。
const INITIAL_ARTICLES_COUNT = 30;

export default async function HomePage() {
  const { contents, totalCount } = await getArticleList({ limit: INITIAL_ARTICLES_COUNT });
  const featured = contents.length > 0 ? await getFeaturedArticle() : null;
  const featuredIds = new Set(featured ? [featured.id] : []);

  const latestDealDate = contents[0]?.dealDate;
  const publishedDates = [...(await getPublishedDates().catch(() => new Set<string>()))];

  // 初回表示分（INITIAL_ARTICLES_COUNT件）のみをItemListとして構造化データ化する。
  // オートスクロールで追加取得される分はクライアント側描画のためJSON-LDには含めない
  // （クロール時点でサーバーが返せる範囲と一致させる）。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${SITE_NAME}｜大量保有・売買の新着開示`,
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
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">
        大量保有報告書で追う日本株の大株主・機関投資家DB
      </h1>
      <p className="mb-2 text-sm leading-relaxed text-ink-secondary">
        EDINETの大量保有報告書とTDnet開示を集計し、企業別の大株主、投資家別の保有銘柄、
        買い増し・売却履歴、その後の株価成績を検索できます。
      </p>
      <div className="relative z-10 mt-5 mb-3 rounded-lg border border-rule bg-section-tint p-4">
        <p className="mb-2 text-sm font-bold text-brand-navy">
          企業名・証券コード・投資家名から検索
        </p>
        <StockSearch prominent />
      </div>
      <nav aria-label="データベースの入口" className="mb-4 flex flex-wrap gap-x-5 gap-y-2 text-sm">
        <Link href="/stocks" className="font-medium text-brand-blue hover:underline">
          日本株・大株主データベース
        </Link>
        <Link href="/investors" className="font-medium text-brand-blue hover:underline">
          機関投資家・大株主データベース
        </Link>
        <Link href="/guides" className="font-medium text-brand-blue hover:underline">
          大量保有報告書の読み方
        </Link>
      </nav>
      {latestDealDate && (
        <DataUpdatedAt
          className="mb-4"
          label="最終更新（反映済みの最新取引日）"
          date={latestDealDate}
          url={SITE_URL}
        />
      )}
      <section className="mb-8" aria-labelledby="database-features">
        <h2 id="database-features" className="mb-3 text-xl font-bold text-brand-navy">
          このデータベースでわかること
        </h2>
        <ul className="grid list-none gap-3 p-0 sm:grid-cols-3">
          <li className="rounded-lg border border-rule bg-paper p-4">
            <Link href="/stocks" className="font-bold text-brand-blue hover:underline">
              企業別の大口保有者
            </Link>
            <p className="mt-1 text-xs leading-relaxed text-ink-secondary">
              5%ルール開示から保有比率と増減履歴を確認
            </p>
          </li>
          <li className="rounded-lg border border-rule bg-paper p-4">
            <Link href="/investors" className="font-bold text-brand-blue hover:underline">
              投資家別の保有銘柄
            </Link>
            <p className="mt-1 text-xs leading-relaxed text-ink-secondary">
              機関投資家・アクティビストの日本株保有を横断
            </p>
          </li>
          <li className="rounded-lg border border-rule bg-paper p-4">
            <Link href="/ranking/returns" className="font-bold text-brand-blue hover:underline">
              開示後の3ヶ月成績
            </Link>
            <p className="mt-1 text-xs leading-relaxed text-ink-secondary">
              買い開示の後に株価がどう動いたかを集計
            </p>
          </li>
        </ul>
      </section>
      {contents.length === 0 ? (
        <p className="text-ink-tertiary">記事がまだありません。</p>
      ) : (
        <>
          {/* 記事一覧より先にランキングの中身を出す。TOPは押した人17.6%で全ページ中最低なのに、
              /trendingは閲覧者全員が押している（2026-08-27のGA4実測）。ヘッダーに同じリンクは
              あるので、足りないのはリンクではなく押す理由＝実際の銘柄名と金額。 */}
          <TopTrendingPreview />
          {featured && (
            <div className="mb-8">
              <FeaturedArticleCard article={featured} rank={1} />
            </div>
          )}
          {/* 一覧は最新の開示日ぶんとは限らない（土日は数日前が最新）ため「今日」と言い切らない。 */}
          <h2 className="mb-4 text-xl font-bold text-brand-navy">大量保有・売買の新着開示</h2>
          <CategoryFilterDetails />
          <InfiniteArticleList
            dateHeadingLevel="h3"
            initialArticles={contents}
            totalCount={totalCount}
            excludeIds={featuredIds}
            publishedDates={publishedDates}
          />
        </>
      )}
    </div>
  );
}
