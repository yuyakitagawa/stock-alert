import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import { formatDate, formatMonth } from "@/lib/format";
import { getAllArticlesForSitemap, getArticlesByDealDate } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

// YYYY-MM-DD形式のみ受け付ける（それ以外はmicroCMSへの無駄な問い合わせをせず404にする）。
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;


// generateStaticParams が無い動的セグメントはNext 16ではリクエスト毎のSSRになり、
// 何度アクセスしてもCDNキャッシュに乗らない（実測: x-vercel-cache: MISS・no-store）。
// 一部でも事前生成しておくとルート全体がISR扱いになり、事前生成していないパラメータも
// 2回目以降はCDNから返る。クロール速度に直結するので主要分だけ事前生成する。
const PRERENDERED_DATES = 60;

export async function generateStaticParams() {
  // 一覧の取得に失敗しても空配列を返してビルドは通す（microCMS/Supabaseの一時障害で
  // デプロイ全体を落とさないため）。空でもルートはISR扱いのままになる。
  try {
    const articles = await getAllArticlesForSitemap();
    const dates: string[] = [];
    for (const article of articles) {
      const date = article.dealDate.slice(0, 10);
      if (!dates.includes(date)) dates.push(date);
      if (dates.length >= PRERENDERED_DATES) break;
    }
    return dates.map((date) => ({ date }));
  } catch {
    return [];
  }
}

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
  const description = `${label}に開示された大量保有・変更報告書をもとにした、大口投資家の動き。全${contents.length}件。`;
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
  // 取引日別ページは日数ぶんだけ増える一方で、/weeklyから張られるのは直近7日分だけで
  // 古い日付は内部リンクが切れていた。月ハブ(/monthly/[month])を親に置いて辿れるようにする。
  const month = date.slice(0, 7);
  const monthLabel = formatMonth(month);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: monthLabel, item: `${SITE_URL}/monthly/${month}` },
      { "@type": "ListItem", position: 3, name: label, item: url },
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
        <Link href={`/monthly/${month}`} className="hover:text-brand-blue">{monthLabel}</Link>
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
      <AdUnit placement="bottom" />
    </div>
  );
}
