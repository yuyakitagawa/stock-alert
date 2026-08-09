import type { Metadata } from "next";
import Link from "next/link";
import { getAllStocksForIndex } from "@/lib/microcms";
import { formatDate } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";

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

export default async function StocksIndexPage() {
  const stocks = await getAllStocksForIndex();

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
      <p className="mb-6 text-sm text-foreground/50">
        {SITE_NAME}が大量保有・自社株買いの動きを追跡している銘柄{stocks.length}件です。
        最終開示日が新しい順に並んでいます。
      </p>
      {stocks.length === 0 ? (
        <p className="text-foreground/50">銘柄データがまだありません。</p>
      ) : (
        <ul className="divide-y divide-rule/50 border-t border-rule">
          {stocks.map((stock) => (
            <li key={stock.stockCode} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-3">
              <Link
                href={`/stocks/${stock.stockCode}`}
                className="font-medium text-brand-blue hover:underline"
              >
                {stock.stockName}（{stock.stockCode}）
              </Link>
              <span className="text-xs text-foreground/40">
                記事{stock.articleCount}件・最終開示{formatDate(stock.latestDealDate)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
