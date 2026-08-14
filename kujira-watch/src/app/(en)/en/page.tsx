import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList, getFeaturedArticles } from "@/lib/microcms";
import { SITE_NAME_EN, SITE_URL } from "@/lib/site";
import { UI } from "@/lib/i18n";

// クローラーが最初のHTML(SSR)だけで辿れるリンク数を増やすため、初回取得件数を
// ARTICLES_PER_PAGE(10件・オートスクロールの追加取得単位)より多めにする。
const INITIAL_ARTICLES_COUNT = 30;

export default async function EnHomePage() {
  const t = UI.en;
  const { contents, totalCount } = await getArticleList({
    limit: INITIAL_ARTICLES_COUNT,
    translatedOnly: true,
  });
  const featuredArticles = contents.length > 0 ? await getFeaturedArticles(20, 3, true) : [];
  const featuredIds = new Set(featuredArticles.map((a) => a.id));

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${SITE_NAME_EN} | Latest Articles`,
    itemListElement: contents.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: article.titleEn ?? article.title,
      url: `${SITE_URL}/en/articles/${article.id}`,
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
      <h1 className="sr-only">{SITE_NAME_EN} | Latest Articles</h1>
      {contents.length === 0 ? (
        <p className="text-foreground/50">{t.noArticlesYet}</p>
      ) : (
        <>
          {featuredArticles.length > 0 && (
            <div className="mb-8 space-y-4">
              {featuredArticles.map((article, i) => (
                <FeaturedArticleCard key={article.id} article={article} rank={i + 1} locale="en" />
              ))}
            </div>
          )}
          <h2 className="mb-4 text-xl font-bold text-brand-navy">{t.todaysDealsHeading}</h2>
          <InfiniteArticleList
            initialArticles={contents}
            totalCount={totalCount}
            excludeIds={featuredIds}
            locale="en"
          />
        </>
      )}
    </div>
  );
}
