import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Box from "@mui/material/Box";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import Typography from "@mui/material/Typography";
import ArticleCard from "@/components/ArticleCard";
import CompanyInfoCard from "@/components/CompanyInfoCard";
import DealDateHeading from "@/components/DealDateHeading";
import DealDateSeeMoreLink from "@/components/DealDateSeeMoreLink";
import DealTypeBadge from "@/components/DealTypeBadge";
import { getCompanyInfo } from "@/lib/companyInfo";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { disclosureDocLabel, edinetPdfUrl } from "@/lib/disclosures";
import { getFilersByStockCode, getHoldingsByStockCode } from "@/lib/investors";
import { formatDate } from "@/lib/format";
import RatioTransition from "@/components/RatioTransition";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { getArticlesByStockCode } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { buildStockDealSummary, formatStockDealSummary } from "@/lib/stockSummary";
import AdUnit from "@/components/AdUnit";
import FaqAccordionList from "@/components/FaqAccordionList";
import { buildStockFaqItems } from "@/lib/stockFaq";

// 会社情報(jpx_stock_list/gen_rankings)はトレーディングシステム側が日次で更新するため、
// microCMS記事(revalidate:60)とずれない範囲で定期的に再取得する。
export const revalidate = 300;

type Props = {
  params: Promise<{ code: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { code } = await params;
  const [{ contents }, { contents: enContents }, companyInfo] = await Promise.all([
    getArticlesByStockCode(code),
    getArticlesByStockCode(code, { translatedOnly: true }),
    getCompanyInfo(code),
  ]);
  const stockName = contents[0]?.stockName ?? companyInfo?.name;
  if (!stockName) return {};

  // 「◯◯ 大量保有」「◯◯ 大株主」の検索意図を狙う（growth_ideas #801-802）。
  const title = `${stockName}（${code}）の大量保有・大株主の動き`;
  const dealSummaryText =
    contents.length > 0
      ? formatStockDealSummary(buildStockDealSummary(contents), stockName, code)
      : `${stockName}（${code}）の大量保有報告書（EDINET）の提出履歴と会社情報。`;
  const description = companyInfo?.description
    ? `${companyInfo.description.replace(/。+$/, "")}。${dealSummaryText}`
    : dealSummaryText;
  const url = `${SITE_URL}/stocks/${code}`;
  const hasEn = enContents.length > 0;

  return {
    title,
    description,
    // 解説記事が1本も無い銘柄は開示テーブルと会社情報だけの薄いページになるため、
    // 検索エンジンには載せない（サイト全体の品質評価を落とさないための保険）。
    // 記事が生成されればそのままindex対象に戻る。
    ...(contents.length === 0 ? { robots: { index: false, follow: true } } : {}),
    alternates: {
      canonical: url,
      ...(hasEn ? { languages: { ja: url, en: `${SITE_URL}/en/stocks/${code}` } } : {}),
    },
    openGraph: { title, description, url },
  };
}

export default async function StockPage({ params }: Props) {
  const { code } = await params;
  const [{ contents }, companyInfo, filers, holdings] = await Promise.all([
    getArticlesByStockCode(code),
    getCompanyInfo(code),
    getFilersByStockCode(code),
    getHoldingsByStockCode(code),
  ]);

  // 解説記事が無くてもEDINET開示・会社情報があれば銘柄ページとして成立させる
  // （記事化は保有比率上位から順に行うため、開示はあるが記事が無い銘柄が大半を占める）。
  // 上場銘柄マスターにも開示にも無いコードだけ404にする。
  const stockName = contents[0]?.stockName ?? companyInfo?.name;
  if (!stockName) {
    notFound();
  }

  const url = `${SITE_URL}/stocks/${code}`;
  const dealSummary = buildStockDealSummary(contents);

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

  const faqItems = buildStockFaqItems(stockName, code, filers, holdings);
  const faqJsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: faqItems.map((faq) => ({
      "@type": "Question",
      name: faq.question,
      acceptedAnswer: { "@type": "Answer", text: faq.answer },
    })),
  };

  return (
    <div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      {contents.length > 0 && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
        />
      )}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">{stockName}（{code}）</span>
      </nav>
      <h1 className={`text-2xl font-bold text-brand-navy sm:text-3xl ${companyInfo?.description ? "mb-2" : "mb-6"}`}>
        {stockName}（{code}）
      </h1>
      {companyInfo?.description && (
        <p className="mb-6 text-sm text-foreground/80">{companyInfo.description}</p>
      )}
      {companyInfo && <CompanyInfoCard info={companyInfo} />}
      {filers.length > 0 && (
        <Box sx={{ mb: 4, pt: 2, borderTop: 1, borderColor: "divider" }}>
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700, color: "primary.main" }}>
            大量保有報告書の提出投資家
          </Typography>
          <List disablePadding dense>
            {filers.map((filer) => (
              <ListItem key={filer.filerName} disableGutters sx={{ py: 0.5, gap: 1, flexWrap: "wrap" }}>
                <Link
                  href={`/investors/${encodeURIComponent(filer.filerName)}`}
                  className="text-brand-blue hover:underline"
                >
                  {filer.filerName}
                </Link>
                <DealTypeBadge dealType={filer.category} />
              </ListItem>
            ))}
          </List>
        </Box>
      )}
      {holdings.length > 0 && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700, color: "primary.main" }}>
            保有比率の推移（EDINET開示ベース）
          </Typography>
          <TableContainer sx={{ borderTop: 1, borderColor: "divider" }}>
            <Table size="small" sx={{ minWidth: 560, "& .MuiTableCell-root": { borderColor: "divider" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: "text.disabled" }}>開示日</TableCell>
                  <TableCell sx={{ color: "text.disabled" }}>投資家</TableCell>
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
                      <Link
                        href={`/investors/${encodeURIComponent(h.filerName)}`}
                        className="text-brand-blue hover:underline"
                      >
                        {h.filerName}
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
        </Box>
      )}
      {contents.length > 0 && (
        <>
          <div className="mb-6">
            <h2 className="text-xl font-bold text-brand-navy">大量保有・自社株買い履歴</h2>
            <p className="mt-1 text-sm text-foreground/80">
              {formatStockDealSummary(dealSummary, stockName, code)}
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
              <DealDateSeeMoreLink date={group.date} />
            </div>
          ))}
        </>
      )}
      <div className="mb-8 border-t border-rule pt-4">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">よくある質問</h2>
        <FaqAccordionList faqs={faqItems} />
      </div>
      <AdUnit placement="bottom" />
    </div>
  );
}
