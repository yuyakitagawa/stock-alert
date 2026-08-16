import type { Metadata } from "next";
import Link from "next/link";
import {
  ACTIVIST_HOLDING_MIN_RATIO,
  getActivistHoldingsSummary,
} from "@/lib/activists";
import { displayFilerName, formatDate } from "@/lib/format";
import { getAllStocksForIndex } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

export const revalidate = 3600;

const url = `${SITE_URL}/activists`;
const title = "アクティビストの保有銘柄一覧";

export const metadata: Metadata = {
  title,
  description:
    "アクティビスト（物言う株主）がEDINET大量保有報告書で開示している現在の保有銘柄をファンド別に一覧。複数のアクティビストが同時に保有する銘柄、保有比率、最終開示日つきで毎日更新しています。",
  alternates: { canonical: url },
  openGraph: { title, url },
};

export default async function ActivistsPage() {
  const [summary, stocksWithArticles] = await Promise.all([
    getActivistHoldingsSummary(),
    getAllStocksForIndex().catch(() => []),
  ]);

  // 銘柄ページ(/stocks/[code])は記事がある銘柄にしか存在しないため、
  // 記事の無い銘柄はリンクにせずテキストのまま出す（/trendingと同じ規律）。
  const codesWithArticles = new Set(stocksWithArticles.map((s) => s.stockCode));

  const stockLabel = (issuerName: string, issuerCode: string) =>
    codesWithArticles.has(issuerCode) ? (
      <Link href={`/stocks/${issuerCode}`} className="text-brand-blue hover:underline">
        {issuerName}（{issuerCode}）
      </Link>
    ) : (
      <span className="text-foreground/80">
        {issuerName}（{issuerCode}）
      </span>
    );

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "アクティビストファンド一覧（現在の保有銘柄つき）",
    // JSON-LDの名称はDB照合が壊れないよう原文（全角）のまま使う（displayFilerNameのdocコメント参照）。
    itemListElement: summary.funds.map((fund, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: fund.filerName,
      url: `${SITE_URL}/investors/${encodeURIComponent(fund.filerName)}`,
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
        <span className="text-foreground/70">{title}</span>
      </nav>

      <div className="mb-8">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}が投資家分類「アクティビスト」（物言う株主）と判定した提出者について、
          EDINET大量保有報告書の銘柄ごとの最新開示を「現在の保有」として集計した一覧です。
          現在{summary.funds.length}ファンドが{summary.stockCount}銘柄（のべ{summary.holdingCount}件）を
          保有しています。最新開示の保有比率が{ACTIVIST_HOLDING_MIN_RATIO}%未満に下がった銘柄は
          報告義務の対象外となりその後の売買が開示されないため、一覧から除外しています。
          開示は義務発生日から最大5営業日遅れる点、その後の売買が反映されていない可能性がある点に
          ご注意ください。
        </p>
      </div>

      {summary.multiHolderStocks.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">
            複数のアクティビストが保有する銘柄
          </h2>
          <p className="mb-4 text-sm text-foreground/60">
            2つ以上のアクティビストファンドが同時に5%以上を保有している銘柄です。
            経営への圧力が強まりやすい状態といえます。
          </p>
          <ul className="border-t border-rule">
            {summary.multiHolderStocks.map((stock) => (
              <li key={stock.issuerCode} className="border-b border-rule py-3 text-sm">
                <span className="font-medium">{stockLabel(stock.issuerName, stock.issuerCode)}</span>
                <span className="ml-2 text-xs text-foreground/50">
                  {stock.holders.length}ファンドが保有
                </span>
                <ul className="mt-1 space-y-0.5 pl-4 text-xs text-foreground/60">
                  {stock.holders.map((holder) => (
                    <li key={holder.filerName}>
                      <Link
                        href={`/investors/${encodeURIComponent(holder.filerName)}`}
                        className="text-brand-blue hover:underline"
                      >
                        {displayFilerName(holder.filerName)}
                      </Link>
                      <span className="ml-2">
                        {holder.holdingRatio}%（{formatDate(holder.discDate)}時点）
                      </span>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="mb-10">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">ファンド別の保有銘柄</h2>
        <p className="mb-4 text-sm text-foreground/60">
          保有銘柄数の多い順です。ファンド名をクリックすると保有比率の推移や乗っかりリターン実績を
          確認できます。
        </p>
        <ul className="border-t border-rule">
          {summary.funds.map((fund) => (
            <li key={fund.filerName} className="border-b border-rule py-3 text-sm">
              <Link
                href={`/investors/${encodeURIComponent(fund.filerName)}`}
                className="font-medium text-brand-blue hover:underline"
              >
                {displayFilerName(fund.filerName)}
              </Link>
              <span className="ml-2 text-xs text-foreground/50">{fund.holdings.length}銘柄</span>
              <ul className="mt-1 space-y-0.5 pl-4 text-xs text-foreground/60">
                {fund.holdings.map((holding) => (
                  <li key={holding.issuerCode}>
                    {stockLabel(holding.issuerName, holding.issuerCode)}
                    <span className="ml-2">
                      {holding.holdingRatio}%（{formatDate(holding.discDate)}時点）
                    </span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      </section>

      <nav className="flex flex-wrap gap-x-6 gap-y-2 border-t border-rule pt-6 text-sm">
        <Link href="/ranking/activist" className="text-brand-blue hover:underline">
          直近30日にアクティビストが動いた銘柄を見る ›
        </Link>
        <Link href="/disclosures" className="text-brand-blue hover:underline">
          開示速報で最新の動きを見る ›
        </Link>
        <Link href="/investors?category=アクティビスト" className="text-brand-blue hover:underline">
          アクティビスト一覧を見る ›
        </Link>
      </nav>
      <AdUnit placement="bottom" />
    </div>
  );
}
