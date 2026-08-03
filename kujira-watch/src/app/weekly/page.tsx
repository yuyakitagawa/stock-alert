import type { Metadata } from "next";
import Link from "next/link";
import ArticleCard from "@/components/ArticleCard";
import DealDateHeading from "@/components/DealDateHeading";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { getRecentArticles } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { formatDate } from "@/lib/format";

// 「大口投資家の動きを教えて」等の包括的な検索・LLMクエリに直答するための集約ページ。
// 直近7日間の開示を横断的に要約する（個別記事は取引ごとの解説に特化しているため、
// このページが唯一の「まとめて見る」導線になる）。
const WINDOW_DAYS = 7;

export async function generateMetadata(): Promise<Metadata> {
  const { contents } = await getRecentArticles(WINDOW_DAYS);
  const title = "大口投資家の動きまとめ（直近7日間）";
  const description = `EDINET大量保有報告書をもとにした、直近${WINDOW_DAYS}日間の大口投資家（クジラ）の動きまとめ。${contents.length}件の開示を取引日順に一覧できます。`;
  const url = `${SITE_URL}/weekly`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function WeeklyDigestPage() {
  const { contents } = await getRecentArticles(WINDOW_DAYS);
  const url = `${SITE_URL}/weekly`;

  const totalAmount = contents.reduce((sum, a) => sum + (a.dealAmount || 0), 0);
  const oldestDate = contents.length > 0 ? contents[contents.length - 1].dealDate : null;
  const newestDate = contents.length > 0 ? contents[0].dealDate : null;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "大口投資家の動きまとめ", item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "大口投資家の動きまとめ（直近7日間）",
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
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-gray-500">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-gray-700">大口投資家の動きまとめ</span>
      </nav>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-brand-navy">大口投資家の動き（直近7日間）</h1>
        {contents.length > 0 && oldestDate && newestDate ? (
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            {SITE_NAME}がEDINET大量保有報告書をもとに集計した、{formatDate(oldestDate)}〜
            {formatDate(newestDate)}の大口投資家（クジラ）の動きです。この期間に{contents.length}
            件の大量保有・変更報告書が開示され、推定取得金額の合計は約
            {totalAmount.toLocaleString("ja-JP")}億円でした。個別の取引は下記の一覧、または各記事で解説しています。
          </p>
        ) : (
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            直近{WINDOW_DAYS}日間はEDINET大量保有報告書の新規開示がありませんでした。
          </p>
        )}
      </div>
      {groupArticlesByDealDate(contents).map((group) => (
        <div key={group.date} className="mb-8">
          <DealDateHeading label={group.label} />
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {group.articles.map((article) => (
              <ArticleCard key={article.id} article={article} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
