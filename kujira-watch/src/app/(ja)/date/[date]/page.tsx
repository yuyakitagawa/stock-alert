import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import ListPageNextStep from "@/components/ListPageNextStep";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import { formatDate, formatMonth } from "@/lib/format";
import { getAllArticlesForSitemap, getArticlesByDealDate } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { PageDatesJsonLd } from "@/components/DataUpdatedAt";
import { isIndexableDatePage } from "@/lib/pageIndexability";
import AdUnit from "@/components/AdUnit";

// YYYY-MM-DD形式のみ受け付ける（それ以外はmicroCMSへの無駄な問い合わせをせず404にする）。
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;


// generateStaticParams が無い動的セグメントはNext 16ではリクエスト毎のSSRになり、
// 何度アクセスしてもCDNキャッシュに乗らない（実測: x-vercel-cache: MISS・no-store）。
// 一部でも事前生成しておくとルート全体がISR扱いになり、事前生成していないパラメータも
// 2回目以降はCDNから返る。クロール速度に直結するので主要分だけ事前生成する。
// 2026-09-03に件数を大きく減らした: 4ルート合計937ページの事前生成にビルド6.6分（1回8分）かかり、
// Proプランのビルド課金（CPU分単価）の主因になっていたため。事前生成から外れたページも
// 初回アクセスでISR生成されてCDNに載る（ルート全体がISR扱い）ので表示は変わらない。
const PRERENDERED_DATES = 7;

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
  if (!isIndexableDatePage(contents.length)) return {};

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

  // 開示が数件しかない日は記事へのリンクが数本並ぶだけで記事本文と内容が重複するため、
  // ページ自体を公開しない（2026-08-29にnoindexから404へ変更。lib/publishedPages.ts）。
  if (!isIndexableDatePage(contents.length)) {
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
      {/* このページの日付＝取引日そのもの。見出しのtime要素と同じ値をWebPageでも宣言する。 */}
      <PageDatesJsonLd url={url} date={date} />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href={`/monthly/${month}`} className="hover:text-brand-blue">{monthLabel}</Link>
        {" / "}
        <span className="text-ink-secondary">{label}</span>
      </nav>
      <div className="mb-6">
        <h1 className="font-serif text-2xl font-bold text-brand-navy sm:text-3xl">
          <time dateTime={date}>{label}</time>の大口投資家の動き
        </h1>
        <p className="mt-1 text-sm text-ink-tertiary">
          この日に開示された大量保有・変更報告書を{contents.length}件まとめています。
        </p>
      </div>
      {(() => {
        const [top, ...rest] = contents;
        return (
          <>
            <FeaturedArticleCard article={top} rank={1} />
            {rest.length > 0 && (
              <ul className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                {rest.map((article) => (
                  <li key={article.id} className="grid">
                    <ArticleCard article={article} />
                  </li>
                ))}
              </ul>
            )}
          </>
        );
      })()}
      {/* 入口の直帰率100%＝この日の記事を見て終わっていた（2026-08-27のGA4実測）。
          同じ月の他の日と、開示を横断して見るページへ送る。 */}
      <ListPageNextStep
        links={[
          { href: `/monthly/${month}`, label: `${monthLabel}の開示一覧` },
          { href: "/trending", label: "銘柄ランキング" },
          { href: "/activists", label: "アクティビスト注目銘柄" },
        ]}
      />
      <AdUnit placement="bottom" />
    </div>
  );
}
