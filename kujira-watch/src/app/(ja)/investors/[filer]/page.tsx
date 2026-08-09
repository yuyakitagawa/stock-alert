import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import DealTypeBadge from "@/components/DealTypeBadge";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { docTypeLabel, getFilerClassification, getFilerHoldings } from "@/lib/investors";
import { formatDate } from "@/lib/format";
import { SITE_URL } from "@/lib/site";

export const revalidate = 300;

type Props = {
  params: Promise<{ filer: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { filer } = await params;
  const filerName = decodeURIComponent(filer);
  const holdings = await getFilerHoldings(filerName);
  if (holdings.length === 0) return {};

  const title = `${filerName}の大量保有報告書・保有銘柄一覧`;
  const description = `${filerName}がEDINET大量保有報告書（5%ルール）で開示した保有銘柄・保有比率の推移を${holdings.length}件まとめました。`;
  const url = `${SITE_URL}/investors/${filer}`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function InvestorPage({ params }: Props) {
  const { filer } = await params;
  const filerName = decodeURIComponent(filer);

  const [holdings, classification] = await Promise.all([
    getFilerHoldings(filerName),
    getFilerClassification(filerName),
  ]);

  if (holdings.length === 0) {
    notFound();
  }

  const url = `${SITE_URL}/investors/${filer}`;
  const category = classification?.category ?? "その他";

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "投資家一覧", item: `${SITE_URL}/investors` },
      { "@type": "ListItem", position: 3, name: filerName, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${filerName}の保有銘柄一覧`,
    itemListElement: holdings.map((h, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: `${h.issuerName}（${h.issuerCode}）`,
      url: `${SITE_URL}/stocks/${h.issuerCode}`,
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
        <Link href="/investors" className="hover:text-brand-blue">投資家一覧</Link>
        {" / "}
        <span className="text-foreground/70">{filerName}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{filerName}</h1>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <DealTypeBadge dealType={category} />
        {classification?.description && (
          <p className="text-sm text-foreground/60">{classification.description}</p>
        )}
      </div>
      {classification?.profile && (
        <div className="mb-8 border-t border-rule pt-4">
          <h2 className="mb-2 text-sm font-bold text-brand-navy">{filerName}について</h2>
          <p className="whitespace-pre-line text-sm leading-relaxed text-foreground/70">
            {classification.profile}
          </p>
        </div>
      )}
      <p className="mb-6 text-sm text-foreground/50">
        EDINET大量保有報告書（5%ルール）にもとづき、{filerName}が開示した保有銘柄・保有比率の推移を
        {holdings.length}件まとめています。個別銘柄の詳しい解説記事は各銘柄ページからご覧いただけます。
      </p>
      <div className="overflow-x-auto border-t border-rule">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-rule text-xs text-foreground/40">
              <th className="py-2 pr-4 font-normal">開示日</th>
              <th className="py-2 pr-4 font-normal">銘柄</th>
              <th className="py-2 pr-4 font-normal">種別</th>
              <th className="py-2 font-normal">保有比率</th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((h) => (
              <tr key={h.docId} className="border-b border-rule/50">
                <td className="py-3 pr-4 whitespace-nowrap text-foreground/60">
                  {formatDate(h.discDate)}
                </td>
                <td className="py-3 pr-4">
                  <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                    {h.issuerName}（{h.issuerCode}）
                  </Link>
                </td>
                <td className="py-3 pr-4 whitespace-nowrap text-foreground/60">
                  {docTypeLabel(h.docTypeCode)}
                </td>
                <td className="py-3 whitespace-nowrap text-foreground/60">
                  {h.holdingRatioPrior !== null ? `${h.holdingRatioPrior}% → ` : ""}
                  {h.holdingRatio !== null ? `${h.holdingRatio}%` : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {category !== "その他" && (
        <p className="mt-6 text-xs text-foreground/40">
          {DEAL_TYPE_DESCRIPTIONS[category]}
        </p>
      )}
    </div>
  );
}
