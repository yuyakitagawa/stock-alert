import type { Metadata } from "next";
import Link from "next/link";
import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";

// 記事の通覧ページ。/articles/[id] は800本以上あるのに一覧が無く、読者は
// トップの新着かカテゴリ・銘柄・投資家の集約ページ経由でしか記事に辿り着けなかった
// （2026-08-27のAdSense再監査で /articles が404であることを指摘された）。
// 初回SSRの実リンク数はカテゴリページと揃える。
const INITIAL_ARTICLES_COUNT = 30;

const title = "記事一覧";
const description =
  "金融庁EDINETの大量保有報告書（5%ルール）とTDnetの自社株買い開示をもとにした解説記事の一覧。機関投資家・アクティビスト・創業家など大口投資家の売買を、開示日の新しい順に掲載しています。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/articles` },
  openGraph: { title, description, url: `${SITE_URL}/articles` },
};

export const revalidate = 60;

export default async function ArticlesPage() {
  const { contents, totalCount } = await getArticleList({ limit: INITIAL_ARTICLES_COUNT });
  const url = `${SITE_URL}/articles`;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${SITE_NAME}の記事一覧`,
    itemListElement: contents.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: article.title,
      url: `${SITE_URL}/articles/${article.id}`,
    })),
  };

  return (
    <div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">{title}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      <p className="mb-6 text-sm leading-relaxed text-foreground/70">
        全{totalCount.toLocaleString()}本。EDINETの大量保有報告書とTDnetの自社株買い開示を
        日次で取得し、誰がどの銘柄をどれだけ売買したかを開示原本にもとづいて解説しています。
        投資家の種類で絞り込む場合はページ上部のカテゴリから、銘柄や投資家から探す場合は
        <Link href="/stocks" className="text-brand-blue hover:underline">銘柄一覧</Link>
        ・
        <Link href="/investors" className="text-brand-blue hover:underline">投資家一覧</Link>
        をご覧ください。
      </p>
      {contents.length === 0 ? (
        <p className="text-foreground/50">記事がまだありません。</p>
      ) : (
        <InfiniteArticleList initialArticles={contents} totalCount={totalCount} />
      )}
    </div>
  );
}
