import type { Metadata } from "next";
import Link from "next/link";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import Typography from "@mui/material/Typography";
import FilterButtonNav from "@/components/FilterButtonNav";
import { getAllStocksForIndex } from "@/lib/microcms";
import { getAllSectorsByCode } from "@/lib/companyInfo";
import { formatDate } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

const title = "銘柄一覧";
const description =
  "EDINET大量保有報告書（5%ルール）・自社株買いなど、大口投資家の動きが開示された銘柄の一覧。銘柄別に保有・取引の履歴を確認できます。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/stocks` },
  openGraph: { title, description, url: `${SITE_URL}/stocks` },
};

type Props = {
  searchParams: Promise<{ sector?: string }>;
};

export default async function StocksIndexPage({ searchParams }: Props) {
  const [{ sector }, stocks, sectorByCode] = await Promise.all([
    searchParams,
    getAllStocksForIndex(),
    getAllSectorsByCode(),
  ]);

  const counts = new Map<string, number>();
  for (const s of stocks) {
    const sec = sectorByCode.get(s.stockCode);
    if (sec) counts.set(sec, (counts.get(sec) ?? 0) + 1);
  }
  const sectors = Array.from(counts.keys()).sort((a, b) => a.localeCompare(b, "ja"));
  const selectedSector = sector && counts.has(sector) ? sector : null;
  const visibleStocks = selectedSector
    ? stocks.filter((s) => sectorByCode.get(s.stockCode) === selectedSector)
    : stocks;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "銘柄一覧", item: `${SITE_URL}/stocks` },
    ],
  };

  return (
    <div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">銘柄一覧</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">銘柄一覧</h1>
      <p className="mb-4 text-sm text-foreground/50">
        {SITE_NAME}が大量保有・自社株買いの動きを追跡している銘柄{stocks.length}件です。
        証券コード順に並んでいます。
      </p>
      {sectors.length > 0 && (
        <FilterButtonNav
          ariaLabel="業種で絞り込む"
          items={[
            {
              href: "/stocks",
              label: `すべて（${stocks.length}件）`,
              selected: selectedSector === null,
            },
            ...sectors.map((sec) => ({
              href: `/stocks?sector=${encodeURIComponent(sec)}`,
              label: `${sec}（${counts.get(sec)}件）`,
              selected: selectedSector === sec,
            })),
          ]}
        />
      )}
      {visibleStocks.length === 0 ? (
        <p className="text-foreground/50">
          {stocks.length === 0 ? "銘柄データがまだありません。" : "該当する銘柄がありません。"}
        </p>
      ) : (
        <List disablePadding sx={{ borderTop: 1, borderColor: "divider" }}>
          {visibleStocks.map((stock) => (
            <ListItem
              key={stock.stockCode}
              disableGutters
              sx={{ py: 1.5, borderBottom: 1, borderColor: "divider", flexWrap: "wrap", columnGap: 1.5, rowGap: 0.5 }}
            >
              <Link href={`/stocks/${stock.stockCode}`} className="font-medium text-brand-blue hover:underline">
                {stock.stockName}（{stock.stockCode}）
              </Link>
              {sectorByCode.get(stock.stockCode) && (
                <Typography variant="caption" sx={{ color: "text.disabled" }}>
                  {sectorByCode.get(stock.stockCode)}
                </Typography>
              )}
              <Typography variant="caption" sx={{ color: "text.disabled" }}>
                記事{stock.articleCount}件・最終開示{formatDate(stock.latestDealDate)}
              </Typography>
            </ListItem>
          ))}
        </List>
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}
