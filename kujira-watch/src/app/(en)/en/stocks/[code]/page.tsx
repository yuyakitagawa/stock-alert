import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import CompanyInfoCard from "@/components/CompanyInfoCard";
import DealDateHeading from "@/components/DealDateHeading";
import DealDateSeeMoreLink from "@/components/DealDateSeeMoreLink";
import { getCompanyInfo } from "@/lib/companyInfo";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { getArticlesByStockCode } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { UI } from "@/lib/i18n";
import { buildStockDealSummary, formatStockDealSummary } from "@/lib/stockSummary";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

type Props = {
  params: Promise<{ code: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { code } = await params;
  const [{ contents }, companyInfo] = await Promise.all([
    getArticlesByStockCode(code, { translatedOnly: true }),
    getCompanyInfo(code),
  ]);
  if (contents.length === 0) return {};

  const stockName = contents[0].stockName;
  const title = `${stockName} (${code})`;
  const dealSummaryText = formatStockDealSummary(buildStockDealSummary(contents), stockName, code, "en");
  const description = companyInfo?.description
    ? `${companyInfo.description}. ${dealSummaryText}`
    : dealSummaryText;
  const url = `${SITE_URL}/en/stocks/${code}`;

  return {
    title,
    description,
    alternates: { canonical: url, languages: { ja: `${SITE_URL}/stocks/${code}`, en: url } },
    openGraph: { title, description, url },
  };
}

export default async function EnStockPage({ params }: Props) {
  const { code } = await params;
  const t = UI.en;
  const [{ contents }, companyInfo] = await Promise.all([
    getArticlesByStockCode(code, { translatedOnly: true }),
    getCompanyInfo(code),
  ]);

  if (contents.length === 0) {
    notFound();
  }

  const stockName = contents[0].stockName;
  const url = `${SITE_URL}/en/stocks/${code}`;
  const dealSummary = buildStockDealSummary(contents);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: t.top, item: `${SITE_URL}/en` },
      { "@type": "ListItem", position: 2, name: `${stockName} (${code})`, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${stockName} (${code}) — ${t.stockHistoryHeading}`,
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
        <span className="text-foreground/70">{stockName} ({code})</span>
      </nav>
      <h1 className={`text-2xl font-bold text-brand-navy sm:text-3xl ${companyInfo?.description ? "mb-2" : "mb-6"}`}>
        {stockName} ({code})
      </h1>
      {companyInfo?.description && (
        <p className="mb-6 text-sm text-foreground/80">{companyInfo.description}</p>
      )}
      {companyInfo && <CompanyInfoCard info={companyInfo} locale="en" />}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-brand-navy">{t.stockHistoryHeading}</h2>
        <p className="mt-1 text-sm text-foreground/80">
          {formatStockDealSummary(dealSummary, stockName, code, "en")}
        </p>
      </div>
      {groupArticlesByDealDate(contents, "en").map((group) => (
        <div key={group.date} className="mb-8">
          <DealDateHeading label={group.label} />
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {group.articles.map((article) => (
              <ArticleCard key={article.id} article={article} locale="en" />
            ))}
          </div>
          <DealDateSeeMoreLink date={group.date} locale="en" />
        </div>
      ))}
      <AdUnit placement="bottom" locale="en" />
    </div>
  );
}
