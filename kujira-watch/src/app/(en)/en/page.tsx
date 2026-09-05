import EnArticleCard from "@/components/EnArticleCard";
import { getTranslatedArticleRefs } from "@/lib/microcms";
import { EN_SITE_URL, SITE_NAME_EN } from "@/lib/en";
import { SITE_URL } from "@/lib/site";

// 英語版のトップ。英訳済み記事（2026-08-29以前に生成したもの）を取引日の新しい順に並べる。
// オートスクロールは付けず、クローラーが最初のHTMLだけで辿れる件数を多めに出す。
export const revalidate = 3600;

const ARTICLES_ON_TOP = 50;

export default async function EnHomePage() {
  const all = await getTranslatedArticleRefs();
  const articles = all.slice(0, ARTICLES_ON_TOP);

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${SITE_NAME_EN} | Latest Articles`,
    itemListElement: articles.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: article.titleEn,
      url: `${EN_SITE_URL}/articles/${article.id}`,
    })),
  };

  return (
    <div>
      {articles.length > 0 && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
        />
      )}
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">
        Who is buying Japanese stocks? Large-shareholding filings, explained in English
      </h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        Investors who cross 5% of a listed Japanese company must file a large-shareholding report with
        EDINET, the Financial Services Agency&apos;s disclosure system. This site reads those filings
        daily and explains who bought or sold what, in plain English. The English edition covers a
        subset of articles; the{" "}
        <a href={SITE_URL} hrefLang="ja" className="text-brand-blue hover:underline">
          Japanese edition
        </a>{" "}
        carries every disclosure, plus stock and investor pages.
      </p>
      <h2 className="mb-4 text-xl font-bold text-brand-navy">Latest trades</h2>
      {articles.length === 0 ? (
        <p className="text-ink-tertiary">No articles yet.</p>
      ) : (
        <ul className="m-0 grid list-none grid-cols-1 gap-4 p-0 sm:grid-cols-2">
          {articles.map((article) => (
            <li key={article.id}>
              <EnArticleCard article={article} headingLevel="h3" />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
