import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import { formatDate } from "@/lib/format";
import { getArticlesByDealDate } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";

// YYYY-MM-DD形式のみ受け付ける（それ以外はmicroCMSへの無駄な問い合わせをせず404にする）。
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

type Props = {
  params: Promise<{ date: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { date } = await params;
  if (!DATE_PATTERN.test(date)) return {};

  const { contents } = await getArticlesByDealDate(date);
  if (contents.length === 0) return {};

  const label = formatDate(date);
  const title = `${label}の大口投資家の動き`;
  const description = `${label}に開示された大量保有・変更報告書をもとにした、大口投資家（クジラ）の動き。全${contents.length}件。`;
  const url = `${SITE_URL}/date/${date}`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function DateArchivePage({ params }: Props) {
  const { date } = await params;
  if (!DATE_PATTERN.test(date)) {
    notFound();
  }

  const { contents } = await getArticlesByDealDate(date);

  if (contents.length === 0) {
    notFound();
  }

  const label = formatDate(date);
  const url = `${SITE_URL}/date/${date}`;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: label, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${label}の大口投資家の動き`,
    itemListElement: contents.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
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
        <span className="text-foreground/70">{label}</span>
      </nav>
      <div className="mb-6">
        <h1 className="font-serif text-2xl font-bold text-brand-navy sm:text-3xl">
          {label}の大口投資家の動き
        </h1>
        <p className="mt-1 text-sm text-foreground/50">
          この日に開示された大量保有・変更報告書を{contents.length}件まとめています。
        </p>
      </div>
      {(() => {
        const [top, ...rest] = contents;
        return (
          <>
            <FeaturedArticleCard article={top} rank={1} />
            {rest.length > 0 && (
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                {rest.map((article) => (
                  <ArticleCard key={article.id} article={article} />
                ))}
              </div>
            )}
          </>
        );
      })()}
    </div>
  );
}
