import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import InfiniteArticleList from "@/components/InfiniteArticleList";
import { getArticleList } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { DEAL_TYPE_EN, EN_SLUG_TO_DEAL_TYPE } from "@/lib/dealTypeInfo";
import { UI } from "@/lib/i18n";

const INITIAL_ARTICLES_COUNT = 30;

export function generateStaticParams() {
  return Object.keys(EN_SLUG_TO_DEAL_TYPE).map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const dealType = EN_SLUG_TO_DEAL_TYPE[slug];
  if (!dealType) return {};

  const label = DEAL_TYPE_EN[dealType].label;
  const title = `${label} — Article List`;
  const description = `Articles about large-holding transactions by ${label} filers.`;
  return {
    title,
    description,
    alternates: {
      canonical: `${SITE_URL}/en/category/${slug}`,
      languages: { ja: `${SITE_URL}/category/${encodeURIComponent(dealType)}`, en: `${SITE_URL}/en/category/${slug}` },
    },
    openGraph: { title, description, url: `${SITE_URL}/en/category/${slug}` },
  };
}

export default async function EnCategoryPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const dealType = EN_SLUG_TO_DEAL_TYPE[slug];

  if (!dealType) {
    notFound();
  }

  const t = UI.en;
  const { contents, totalCount } = await getArticleList({
    dealType,
    limit: INITIAL_ARTICLES_COUNT,
    translatedOnly: true,
  });
  const url = `${SITE_URL}/en/category/${slug}`;
  const label = DEAL_TYPE_EN[dealType].label;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: t.top, item: `${SITE_URL}/en` },
      { "@type": "ListItem", position: 2, name: label, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${label} — Article List`,
    itemListElement: contents.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: article.titleEn ?? article.title,
      url: `${SITE_URL}/en/articles/${article.id}`,
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
      <nav aria-label={t.breadcrumbAria} className="mb-4 text-xs text-foreground/50">
        <Link href="/en" className="hover:text-brand-blue">{t.top}</Link>
        {" / "}
        <span className="text-foreground/70">{label}</span>
      </nav>
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">
        {t.categoryPrefix}
        {label}
      </h1>
      {contents.length === 0 ? (
        <p className="text-foreground/50">{t.noArticlesInCategory}</p>
      ) : (
        <InfiniteArticleList
          initialArticles={contents}
          totalCount={totalCount}
          dealType={dealType}
          locale="en"
        />
      )}
    </div>
  );
}
