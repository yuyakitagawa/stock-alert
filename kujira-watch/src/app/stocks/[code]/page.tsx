import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import CompanyInfoCard from "@/components/CompanyInfoCard";
import DealDateHeading from "@/components/DealDateHeading";
import { getCompanyInfo } from "@/lib/companyInfo";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { getArticlesByStockCode } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";

// 会社情報(jpx_stock_list/gen_rankings)はトレーディングシステム側が日次で更新するため、
// microCMS記事(revalidate:60)とずれない範囲で定期的に再取得する。
export const revalidate = 300;

type Props = {
  params: Promise<{ code: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { code } = await params;
  const { contents } = await getArticlesByStockCode(code);
  if (contents.length === 0) return {};

  const stockName = contents[0].stockName;
  const title = `${stockName}（${code}）`;
  const description = `${stockName}（${code}）に関する機関投資家・インサイダー・自社株買いなど「クジラ」の動きをまとめました。全${contents.length}件。`;
  const url = `${SITE_URL}/stocks/${code}`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function StockPage({ params }: Props) {
  const { code } = await params;
  const [{ contents }, companyInfo] = await Promise.all([
    getArticlesByStockCode(code),
    getCompanyInfo(code),
  ]);

  if (contents.length === 0) {
    notFound();
  }

  const stockName = contents[0].stockName;
  const url = `${SITE_URL}/stocks/${code}`;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: `${stockName}（${code}）`, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${stockName}（${code}）の大量保有・自社株買い履歴`,
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
        <span className="text-foreground/70">{stockName}（{code}）</span>
      </nav>
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">
        {stockName}（{code}）
      </h1>
      {companyInfo && <CompanyInfoCard info={companyInfo} />}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-brand-navy">大量保有・自社株買い履歴</h2>
        <p className="mt-1 text-sm text-foreground/50">
          機関投資家・インサイダー・自社株買いなど、この銘柄に関する「クジラ」の動きを{contents.length}件まとめています。
        </p>
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
