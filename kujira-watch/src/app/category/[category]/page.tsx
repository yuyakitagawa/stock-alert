import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList } from "@/lib/microcms";
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
}: {
  params: Promise<{ category: string }>;
}) {
  const { category } = await params;
  const decodedCategory = decodeURIComponent(category);
  const dealType = DEAL_TYPE_BY_CATEGORY[decodedCategory];

  if (!dealType) {
    notFound();
  }

  const { contents, totalCount } = await getArticleList({ dealType });
  const url = `${SITE_URL}/category/${category}`;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: decodedCategory, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${decodedCategory}の記事一覧`,
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
        <span className="text-foreground/70">{decodedCategory}</span>
      </nav>
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">
        カテゴリ: {decodedCategory}
      </h1>
      {contents.length === 0 ? (
        <p className="text-foreground/50">このカテゴリの記事がまだありません。</p>
      ) : (
        <InfiniteArticleList
          initialArticles={contents}
          totalCount={totalCount}
          dealType={dealType}
        />
      )}
    </div>
  );
}
