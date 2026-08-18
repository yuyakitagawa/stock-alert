import type { Metadata } from "next";
import Link from "next/link";
import {
  getActivistHoldingsSummary,
  getActivistRecentMoves,
  type ActivistMove,
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
    "アクティビスト（物言う株主）が直近30日に大きく買い入れた注目銘柄と、EDINET大量保有報告書による直近の動き（買い増し・売却）を一覧。保有比率・開示日つきで毎日更新しています。",
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

  const moveItem = (move: ActivistMove) => (
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
  );

  // 注目銘柄=直近30日にアクティビストが保有比率を増やした（買い入れた）銘柄。
  // 銘柄ごとに増加幅(pt)を合算し、大きい順に並べる。前回比率が無い開示は新規保有とみなし
  // 今回比率ぶんを増加として扱う。複数ファンドが同時保有中の銘柄にはバッジを添える。
  type AttentionRow = {
    issuerCode: string;
    issuerName: string;
    totalDelta: number;
    buys: ActivistMove[];
    multiHolder: boolean;
  };
  const multiHolderCodes = new Set(summary.multiHolderStocks.map((s) => s.issuerCode));
  const byIssuer = new Map<string, AttentionRow>();
  for (const move of recentMoves) {
    if (move.holdingRatio === null) continue;
    const delta =
      move.holdingRatioPrior === null
        ? move.holdingRatio
        : Math.round((move.holdingRatio - move.holdingRatioPrior) * 100) / 100;
    if (delta <= 0) continue;
    const row = byIssuer.get(move.issuerCode) ?? {
      issuerCode: move.issuerCode,
      issuerName: move.issuerName,
      totalDelta: 0,
      buys: [],
      multiHolder: multiHolderCodes.has(move.issuerCode),
    };
    row.totalDelta = Math.round((row.totalDelta + delta) * 100) / 100;
    row.buys.push(move);
    byIssuer.set(move.issuerCode, row);
  }
  const attentionStocks = [...byIssuer.values()].sort((a, b) => b.totalDelta - a.totalDelta);

  const attentionItem = (row: AttentionRow) => (
    <li key={row.issuerCode} className="border-b border-rule py-3 text-sm">
      <span className="font-medium">{stockLabel(row.issuerName, row.issuerCode)}</span>
      <span className="ml-2 text-xs font-bold text-emerald-700">▲{row.totalDelta}pt買い入れ</span>
      {row.multiHolder && (
        <span className="kicker ml-2 whitespace-nowrap text-brand-gold">複数ファンド保有</span>
      )}
      <ul className="mt-1 space-y-0.5 pl-4 text-xs text-foreground/60">
        {row.buys.map((move) => (
          <li key={move.docId}>
            <Link
              href={`/investors/${encodeURIComponent(move.filerName)}`}
              className="text-brand-blue hover:underline"
            >
              {displayFilerName(move.filerName)}
            </Link>
            <span className="ml-2">
              {move.holdingRatioPrior === null && "新規 "}
              <RatioTransition ratio={move.holdingRatio} prior={move.holdingRatioPrior} />
              （{formatDate(move.discDate)}）
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
        </p>
      </div>

      {attentionStocks.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">アクティビスト注目銘柄</h2>
          <p className="mb-4 text-sm text-foreground/60">
            直近{MOVES_WINDOW_DAYS}日にアクティビストが保有比率を増やした（買い入れた）銘柄を、
            増加幅の合計が大きい順に表示しています。詳しくは
            <Link href="/faq/usage" className="text-brand-blue hover:underline">
              よくある質問
            </Link>
            をご覧ください。
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
        <h2 className="mb-2 text-lg font-bold text-brand-navy">直近のアクティビストの動き</h2>
        <p className="mb-4 text-sm text-foreground/60">
          アクティビストが直近{MOVES_WINDOW_DAYS}日に提出した大量保有・変更報告書を新しい順に
          一覧しています。保有比率が増えた開示は買い増し、減った開示は売却方向の動きです。
        </p>
        {recentMoves.length === 0 ? (
          <p className="text-sm text-foreground/60">
            直近{MOVES_WINDOW_DAYS}日にアクティビストの開示はありません。
          </p>
        ) : (
          <>
            <ul className="border-t border-rule">
              {recentMoves.slice(0, MOVES_DISPLAY_LIMIT).map(moveItem)}
            </ul>
            {recentMoves.length > MOVES_DISPLAY_LIMIT && (
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-brand-blue hover:underline">
                  もっと見る（残り{recentMoves.length - MOVES_DISPLAY_LIMIT}件）
                </summary>
                <ul className="mt-2 border-t border-rule">
                  {recentMoves.slice(MOVES_DISPLAY_LIMIT).map(moveItem)}
                </ul>
              </details>
            )}
          </>
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
