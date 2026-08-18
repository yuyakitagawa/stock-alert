import type { Metadata } from "next";
import Link from "next/link";
import {
  ACTIVIST_HOLDING_MIN_RATIO,
  getActivistHoldingsSummary,
  getActivistRecentMoves,
  type ActivistTargetStock,
} from "@/lib/activists";
import { displayFilerName, formatDate } from "@/lib/format";
import { getAllStocksForIndex } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import RatioTransition from "@/components/RatioTransition";
import AdUnit from "@/components/AdUnit";

export const revalidate = 3600;

// 「直近の動き」の集計期間（/ranking/activist・/trendingと同じ30日）。
const MOVES_WINDOW_DAYS = 30;
// 30日分は150件超になり下のセクションが埋もれるため、表示は最新20件に絞る。
const MOVES_DISPLAY_LIMIT = 20;
// アクティビスト注目銘柄の初期表示件数。残りは「もっと見る」で開く。
const ATTENTION_DISPLAY_LIMIT = 10;

const url = `${SITE_URL}/activists`;
const title = "アクティビストの動き";

export const metadata: Metadata = {
  title,
  description:
    "アクティビスト（物言う株主）が複数同時に保有する注目銘柄と、EDINET大量保有報告書による直近の動き（買い増し・売却）を一覧。保有比率・最終開示日つきで毎日更新しています。",
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

  // 注目銘柄は「動きがあった順」＝保有ファンドの最新開示日が新しい順に並べる。
  const latestMoveDate = (stock: ActivistTargetStock) =>
    stock.holders.reduce((max, h) => (h.discDate > max ? h.discDate : max), "");
  const attentionStocks = [...summary.multiHolderStocks].sort((a, b) =>
    latestMoveDate(b).localeCompare(latestMoveDate(a))
  );

  const attentionItem = (stock: ActivistTargetStock) => (
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
  );

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
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
        <span className="text-foreground/70">{title}</span>
      </nav>

      <div className="mb-8">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}が投資家分類「アクティビスト」（物言う株主）と判定した提出者の、
          注目銘柄とEDINET大量保有報告書による直近の動き（買い増し・売却）です。
          現在{summary.funds.length}ファンドが{summary.stockCount}銘柄（のべ{summary.holdingCount}件）を
          保有しています。
          ※開示は最大5営業日遅れて提出される点にご注意ください。
        </p>
      </div>

      {attentionStocks.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">アクティビスト注目銘柄</h2>
          <p className="mb-4 text-sm text-foreground/60">
            2つ以上のアクティビストファンドが、最新開示ベースで同時に
            {ACTIVIST_HOLDING_MIN_RATIO}%以上を保有している銘柄です。動きがあった
            （保有ファンドの開示が新しい）順に表示しています。経営への圧力が強まりやすい
            状態といえます。各ファンドの保有銘柄一覧はファンド名のリンク先（投資家ページ）で
            ご覧いただけます。
          </p>
          <ul className="border-t border-rule">
            {attentionStocks.slice(0, ATTENTION_DISPLAY_LIMIT).map(attentionItem)}
          </ul>
          {attentionStocks.length > ATTENTION_DISPLAY_LIMIT && (
            <details className="mt-3">
              <summary className="cursor-pointer text-sm text-brand-blue hover:underline">
                もっと見る（残り{attentionStocks.length - ATTENTION_DISPLAY_LIMIT}銘柄）
              </summary>
              <ul className="mt-2 border-t border-rule">
                {attentionStocks.slice(ATTENTION_DISPLAY_LIMIT).map(attentionItem)}
              </ul>
            </details>
          )}
        </section>
      )}

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
