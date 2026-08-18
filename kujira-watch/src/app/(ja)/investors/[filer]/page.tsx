import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import DealTypeBadge from "@/components/DealTypeBadge";
import RatioTransition from "@/components/RatioTransition";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { disclosureDocLabel, edinetPdfUrl } from "@/lib/disclosures";
import { getFilerClassification, getFilerHoldings } from "@/lib/investors";
import { displayFilerName, formatDate } from "@/lib/format";
import { SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";
import FaqAccordionList from "@/components/FaqAccordionList";
import { buildInvestorFaqItems } from "@/lib/investorFaq";

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

  // holdingsはdisc_date降順のため、issuerCodeごとに最初に出現したもの(=最新の開示)を採用する。
  const majorHoldings = Array.from(
    new Map(holdings.map((h) => [h.issuerCode, h])).values()
  );
  // 最新開示の増減方向で「買い増し・新規」「売却」に分ける（前回比率が取れない開示は方向不明として除外）。
  const direction = (h: (typeof majorHoldings)[number]) =>
    h.holdingRatio !== null && h.holdingRatioPrior !== null
      ? h.holdingRatio - h.holdingRatioPrior
      : null;
  const recentBuys = majorHoldings.filter((h) => {
    const d = direction(h);
    return d !== null && d > 0;
  });
  const recentSells = majorHoldings.filter((h) => {
    const d = direction(h);
    return d !== null && d < 0;
  });

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

  const faqItems = buildInvestorFaqItems(filerName, majorHoldings, recentBuys);
  const faqJsonLd =
    faqItems.length > 0
      ? {
          "@context": "https://schema.org",
          "@type": "FAQPage",
          mainEntity: faqItems.map((faq) => ({
            "@type": "Question",
            name: faq.question,
            acceptedAnswer: { "@type": "Answer", text: faq.answer },
          })),
        }
      : null;

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
      {faqJsonLd && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
        />
      )}
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href="/investors" className="hover:text-brand-blue">投資家一覧</Link>
        {" / "}
        <span className="text-foreground/70">{displayFilerName(filerName)}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{displayFilerName(filerName)}</h1>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <DealTypeBadge dealType={category} />
        {classification?.description && (
          <p className="text-sm text-foreground/60">{classification.description}</p>
        )}
      </div>
      {classification?.profile && (
        <div className="mb-8 border-t border-rule pt-4">
          <h2 className="mb-2 text-sm font-bold text-brand-navy">{displayFilerName(filerName)}について</h2>
          <p className="whitespace-pre-line text-sm leading-relaxed text-foreground/70">
            {classification.profile}
          </p>
        </div>
      )}
      {majorHoldings.length > 0 && (
        <div className="mb-8 border-t border-rule pt-4">
          <h2 className="mb-2 text-sm font-bold text-brand-navy">主な保有銘柄</h2>
          <ul className="space-y-2 text-sm">
            {majorHoldings.map((h) => (
              <li key={h.issuerCode}>
                <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                  {h.issuerName}（{h.issuerCode}）
                </Link>
                {h.holdingRatio !== null && (
                  <span className="ml-2 text-xs text-foreground/40">保有比率{h.holdingRatio}%</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {(recentBuys.length > 0 || recentSells.length > 0) && (
        <div className="mb-8 grid grid-cols-1 gap-6 border-t border-rule pt-4 sm:grid-cols-2">
          {recentBuys.length > 0 && (
            <div>
              <h2 className="mb-2 text-sm font-bold text-brand-navy">直近で買い増した銘柄</h2>
              <ul className="space-y-2 text-sm">
                {recentBuys.map((h) => (
                  <li key={h.issuerCode}>
                    <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                      {h.issuerName}（{h.issuerCode}）
                    </Link>
                    <span className="ml-2 text-xs">
                      <RatioTransition ratio={h.holdingRatio} prior={h.holdingRatioPrior} />
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {recentSells.length > 0 && (
            <div>
              <h2 className="mb-2 text-sm font-bold text-brand-navy">直近で売却した銘柄</h2>
              <ul className="space-y-2 text-sm">
                {recentSells.map((h) => (
                  <li key={h.issuerCode}>
                    <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                      {h.issuerName}（{h.issuerCode}）
                    </Link>
                    <span className="ml-2 text-xs">
                      <RatioTransition ratio={h.holdingRatio} prior={h.holdingRatioPrior} />
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      <div className="mb-6 border-t border-rule pt-4">
        <h2 className="text-xl font-bold text-brand-navy">最近の取引</h2>
        <p className="mt-1 text-sm text-foreground/50">
          EDINET大量保有報告書（5%ルール）にもとづき、{filerName}が開示した保有銘柄・保有比率の推移を
          {holdings.length}件まとめています。個別銘柄の詳しい解説記事は各銘柄ページからご覧いただけます。
        </p>
      </div>
      <TableContainer sx={{ borderTop: 1, borderColor: "divider" }}>
        <Table size="small" sx={{ minWidth: 560, "& .MuiTableCell-root": { borderColor: "divider" } }}>
          <TableHead>
            <TableRow>
              <TableCell sx={{ color: "text.disabled" }}>開示日</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>銘柄</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>種別</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>保有比率</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>原文</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {holdings.map((h) => (
              <TableRow key={h.docId}>
                <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                  {formatDate(h.discDate)}
                </TableCell>
                <TableCell>
                  <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                    {h.issuerName}（{h.issuerCode}）
                  </Link>
                </TableCell>
                <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                  {disclosureDocLabel(h)}
                </TableCell>
                <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                  <RatioTransition ratio={h.holdingRatio} prior={h.holdingRatioPrior} />
                </TableCell>
                {/* 一次ソース（EDINETの原文PDF）への直リンク。以前は/disclosuresだけが
                    持っていたが、同ページ廃止にあたり銘柄・投資家ページへ移した。 */}
                <TableCell sx={{ whiteSpace: "nowrap" }}>
                  <a
                    href={edinetPdfUrl(h.docId)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-brand-blue hover:underline"
                  >
                    PDF ↗
                  </a>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      {category !== "その他" && (
        <p className="mt-6 text-xs text-foreground/40">
          {DEAL_TYPE_DESCRIPTIONS[category]}
        </p>
      )}
      {faqItems.length > 0 && (
        <div className="mt-8 border-t border-rule pt-4">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">よくある質問</h2>
          <FaqAccordionList faqs={faqItems} />
        </div>
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}
