import type { Metadata } from "next";
import Link from "next/link";
import {
  ACTIVIST_HOLDING_MIN_RATIO,
  getActivistHoldingsSummary,
  getActivistRecentMoves,
} from "@/lib/activists";
import { displayFilerName, formatDate } from "@/lib/format";
import { getAllStocksForIndex } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import RatioTransition from "@/components/RatioTransition";
import AdUnit from "@/components/AdUnit";

export const revalidate = 3600;

// 「直近の動き」の集計期間（/ranking/activist・/trendingと同じ30日）。
const MOVES_WINDOW_DAYS = 30;
// 30日分は150件超になり保有一覧が埋もれるため、表示は最新50件に絞る。
const MOVES_DISPLAY_LIMIT = 50;

const url = `${SITE_URL}/activists`;
const title = "アクティビストの動き";

export const metadata: Metadata = {
  title,
  description:
    "アクティビスト（物言う株主）のEDINET大量保有報告書による直近の動き（買い増し・売却）と、現在の保有銘柄をファンド別に一覧。複数のアクティビストが同時に保有する銘柄、保有比率、最終開示日つきで毎日更新しています。",
  alternates: { canonical: url },
  openGraph: { title, url },
};

export default async function ActivistsPage() {
  const [summary, recentMoves, stocksWithArticles] = await Promise.all([
    getActivistHoldingsSummary(),
    getActivistRecentMoves(MOVES_WINDOW_DAYS).catch(() => []),
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
          {SITE_NAME}が投資家分類「アクティビスト」（物言う株主）と判定した提出者の、
          EDINET大量保有報告書による直近の動き（買い増し・売却）と現在の保有銘柄です。
          現在{summary.funds.length}ファンドが{summary.stockCount}銘柄（のべ{summary.holdingCount}件）を
          保有しています。
          ※開示は最大5営業日遅れて提出される点にご注意ください。
        </p>
      </div>

      <section className="mb-10">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">
          直近{MOVES_WINDOW_DAYS}日の動き
        </h2>
        <p className="mb-4 text-sm text-foreground/60">
          アクティビストが提出した大量保有・変更報告書を新しい順に一覧しています。
          保有比率が増えた開示は買い増し、減った開示は売却方向の動きです。
        </p>
        {recentMoves.length === 0 ? (
          <p className="text-sm text-foreground/60">
            直近{MOVES_WINDOW_DAYS}日にアクティビストの開示はありません。
          </p>
        ) : (
          <ul className="border-t border-rule">
            {recentMoves.slice(0, MOVES_DISPLAY_LIMIT).map((move) => (
              <li
                key={move.docId}
                className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-rule py-2.5 text-sm"
              >
                <span className="kicker whitespace-nowrap text-foreground/50">
                  {formatDate(move.discDate)}
                </span>
                <Link
                  href={`/investors/${encodeURIComponent(move.filerName)}`}
                  className="text-brand-blue hover:underline"
                >
                  {displayFilerName(move.filerName)}
                </Link>
                <span className="font-medium">{stockLabel(move.issuerName, move.issuerCode)}</span>
                <span className="text-foreground/70">
                  <RatioTransition ratio={move.holdingRatio} prior={move.holdingRatioPrior} />
                </span>
              </li>
            ))}
          </ul>
        )}
        {recentMoves.length > MOVES_DISPLAY_LIMIT && (
          <p className="mt-2 text-xs text-foreground/50">
            ※直近{MOVES_WINDOW_DAYS}日の全{recentMoves.length}件のうち最新{MOVES_DISPLAY_LIMIT}件を
            表示しています。すべての開示は
            <Link href="/disclosures" className="text-brand-blue hover:underline">開示速報</Link>
            でご覧いただけます。
          </p>
        )}
      </section>

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
          銘柄ごとの最新開示を「現在の保有」として集計し、保有銘柄数の多い順に並べています。
          最新開示の保有比率が{ACTIVIST_HOLDING_MIN_RATIO}%未満に下がった銘柄は報告義務の対象外となり
          その後の売買が開示されないため、一覧から除外しています。ファンド名をクリックすると
          保有比率の推移や乗っかりリターン実績を確認できます。
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
