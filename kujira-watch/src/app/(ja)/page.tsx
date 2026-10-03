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
import {
  editableAttrs,
  getStoryContent,
  isStoryblokDraft,
  safeHref,
  textField,
  type StoryblokBlok,
} from "@/lib/storyblok";

// クローラーが最初のHTML(SSR)だけで辿れるリンク数を増やすため、初回取得件数を
// ARTICLES_PER_PAGE(10件・オートスクロールの追加取得単位)より多めにする。
// オートスクロールはJSでのみ発火するため、初回SSR分の実リンクがクロール可能な記事数の下限になる。
const INITIAL_ARTICLES_COUNT = 30;

const DEFAULT_HEADLINE = "大量保有報告書で追う日本株の大株主・機関投資家DB";
const DEFAULT_LEAD =
  "EDINETの大量保有報告書とTDnet開示を集計し、企業別の大株主、投資家別の保有銘柄、買い増し・売却履歴、その後の株価成績を検索できます。";

// TOPの見出し・リード文・セクションの並び順はStoryblokのストーリー「home」で編集する。
// セクションの中身（ランキング・記事一覧など）はデータから描くので、CMSが持つのは並び順と
// 見出しの文言だけ。ストーリーが無い・取得に失敗した時は、この既定の並びで描く。
const DEFAULT_SECTIONS: StoryblokBlok[] = [
  { _uid: "default-features", component: "top_features" },
  { _uid: "default-trending", component: "top_trending" },
  { _uid: "default-featured", component: "top_featured" },
  { _uid: "default-latest", component: "top_latest" },
];

function resolveSections(home: StoryblokBlok | null): StoryblokBlok[] {
  const known = new Set([...DEFAULT_SECTIONS.map((b) => b.component), "notice"]);
  const body = (Array.isArray(home?.body) ? (home.body as StoryblokBlok[]) : []).filter((b) =>
    known.has(b.component)
  );
  if (body.length === 0) return DEFAULT_SECTIONS;
  // 新着一覧はSSRで30件の記事リンクを出す、クロールの入口そのもの（上の注記参照）。
  // 編集でうっかり消してもTOPから記事へのリンクが消えないよう、無ければ末尾に足す。
  return body.some((b) => b.component === "top_latest")
    ? body
    : [...body, DEFAULT_SECTIONS[DEFAULT_SECTIONS.length - 1]];
}

// 編集者が自由に足せるお知らせ枠。本文はプレーンテキストとして描画する（HTMLは通さない）。
function Notice({ blok, draft }: { blok: StoryblokBlok; draft: boolean }) {
  const title = textField(blok, "title");
  const text = textField(blok, "text");
  const href = safeHref(textField(blok, "link_url"));
  const linkLabel = textField(blok, "link_label") || "詳しく見る";
  if (!title && !text) return null;
  const accent = blok.tone === "gold" ? "border-l-brand-gold" : "border-l-brand-blue";
  return (
    <aside
      {...editableAttrs(blok, draft)}
      className={`mb-8 rounded-lg border border-l-4 border-rule ${accent} bg-paper p-4`}
    >
      {title && <p className="mb-1 font-bold text-brand-navy">{title}</p>}
      {text && <p className="whitespace-pre-line text-sm leading-relaxed text-ink-secondary">{text}</p>}
      {href && (
        <a href={href} className="mt-2 inline-block text-sm font-bold text-brand-blue hover:underline">
          {linkLabel} →
        </a>
      )}
    </aside>
  );
}

export default async function HomePage() {
  const [{ contents, totalCount }, home, draft] = await Promise.all([
    getArticleList({ limit: INITIAL_ARTICLES_COUNT }),
    getStoryContent("home"),
    isStoryblokDraft(),
  ]);
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
      <div {...editableAttrs(home, draft)}>
        <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">
          {textField(home, "headline") || DEFAULT_HEADLINE}
        </h1>
        <p className="mb-2 text-sm leading-relaxed text-ink-secondary">
          {textField(home, "lead") || DEFAULT_LEAD}
        </p>
      </div>
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
      {resolveSections(home).map((blok) => {
        const attrs = editableAttrs(blok, draft);
        const heading = textField(blok, "heading");
        switch (blok.component) {
          case "top_features":
            return (
              <section key={blok._uid} className="mb-8" aria-labelledby="database-features" {...attrs}>
                <h2 id="database-features" className="mb-3 text-xl font-bold text-brand-navy">
                  {heading || "このデータベースでわかること"}
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
            );
          // TOPは押した人17.6%で全ページ中最低なのに、/trendingは閲覧者全員が押している
          // （2026-08-27のGA4実測）。足りないのはリンクではなく押す理由＝実際の銘柄名と金額なので、
          // 既定の並びでは記事一覧より先に置く。
          case "top_trending":
            return contents.length > 0 ? (
              <div key={blok._uid} {...attrs}>
                <TopTrendingPreview />
              </div>
            ) : null;
          case "top_featured":
            return featured ? (
              <div key={blok._uid} className="mb-8" {...attrs}>
                <FeaturedArticleCard article={featured} rank={1} />
              </div>
            ) : null;
          case "top_latest":
            return (
              <section key={blok._uid} {...attrs}>
                {/* 一覧は最新の開示日ぶんとは限らない（土日は数日前が最新）ため「今日」と言い切らない。 */}
                <h2 className="mb-4 text-xl font-bold text-brand-navy">
                  {heading || "大量保有・売買の新着開示"}
                </h2>
                {contents.length === 0 ? (
                  <p className="text-ink-tertiary">記事がまだありません。</p>
                ) : (
                  <>
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
              </section>
            );
          case "notice":
            return <Notice key={blok._uid} blok={blok} draft={draft} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
