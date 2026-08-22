import type { Metadata } from "next";
import Link from "next/link";
import {
  getActivistHoldingsSummary,
  getActivistRecentMoves,
  type ActivistMove,
} from "@/lib/activists";
import { displayFilerName, formatDate } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import RatioTransition from "@/components/RatioTransition";
import AdUnit from "@/components/AdUnit";
import { getAllListedCodes } from "@/lib/companyInfo";
import { getFilerIdMap, investorPath } from "@/lib/investors";

export const revalidate = 3600;

// 買い入れの集計期間（/ranking/activist・/trendingと同じ30日）。
const MOVES_WINDOW_DAYS = 30;
// アクティビスト注目銘柄の表示件数。全件を一度に並べるが、30日分を全部描画すると
// ページのHTMLが重くなるため件数自体をここで打ち切る
// （打ち切った件数は画面にも明記して、全件があるように見せない）。
const ATTENTION_RENDER_LIMIT = 100;

const url = `${SITE_URL}/activists`;
const title = "アクティビスト注目銘柄";

export const metadata: Metadata = {
  title,
  description:
    "アクティビスト（物言う株主）が直近30日に大きく買い入れた注目銘柄を、EDINET大量保有報告書をもとに増加幅の大きい順で一覧。保有比率・開示日つきで毎日更新しています。",
  alternates: { canonical: url },
  openGraph: { title, url },
};

export default async function ActivistsPage() {
  const [summary, recentMoves, listedCodes, filerIds] = await Promise.all([
    getActivistHoldingsSummary(),
    getActivistRecentMoves(MOVES_WINDOW_DAYS).catch(() => []),
    getAllListedCodes().catch(() => new Set<string>()),
    getFilerIdMap().catch(() => ({}) as Record<string, number>),
  ]);

  // 銘柄ページ(/stocks/[code])は上場銘柄マスターに載っていれば解説記事が無くても
  // 開示履歴＋会社情報で成立する。マスターに無いコード（上場廃止等）だけ
  // リンクにせずテキストのまま出す（404へのリンクを作らない。/trendingと同じ規律）。

  const stockLabel = (issuerName: string, issuerCode: string) =>
    listedCodes.has(issuerCode) ? (
      <Link href={`/stocks/${issuerCode}`} className="text-brand-blue hover:underline">
        {issuerName}（{issuerCode}）
      </Link>
    ) : (
      <span className="text-foreground/80">
        {issuerName}（{issuerCode}）
      </span>
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
  const attentionStocksAll = [...byIssuer.values()].sort((a, b) => b.totalDelta - a.totalDelta);
  const attentionStocks = attentionStocksAll.slice(0, ATTENTION_RENDER_LIMIT);
  const attentionOmitted = attentionStocksAll.length - attentionStocks.length;

  const attentionItem = (row: AttentionRow) => (
    <li key={row.issuerCode} className="card text-sm">
      <span className="font-medium">{stockLabel(row.issuerName, row.issuerCode)}</span>
      <span className="ml-2 whitespace-nowrap text-xs font-bold text-gain">▲{row.totalDelta}pt買い入れ</span>
      {row.multiHolder && (
        <span className="kicker ml-2 whitespace-nowrap text-brand-gold">複数ファンド保有</span>
      )}
      <ul className="mt-1 space-y-0.5 pl-4 text-xs text-foreground/60">
        {row.buys.map((move) => (
          <li key={move.docId}>
            <Link
              href={investorPath(filerIds[move.filerName], move.filerName)}
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
          {SITE_NAME}が投資家分類「アクティビスト」（物言う株主）と判定した提出者が、
          直近{MOVES_WINDOW_DAYS}日に保有比率を増やした（買い入れた）銘柄を、
          増加幅の合計が大きい順に表示しています。出典はEDINET大量保有報告書です。詳しくは
          <Link href="/faq/usage" className="text-brand-blue hover:underline">
            よくある質問
          </Link>
          をご覧ください。
          {attentionOmitted > 0 && `（増加幅の大きい上位${ATTENTION_RENDER_LIMIT}銘柄を表示。ほか${attentionOmitted}銘柄は各銘柄ページでご確認ください）`}
        </p>
      </div>

      <section className="mb-10">
        {attentionStocks.length === 0 ? (
          <p className="text-sm text-foreground/60">
            直近{MOVES_WINDOW_DAYS}日にアクティビストが買い入れた銘柄はありません。
          </p>
        ) : (
          <ul className="card-grid card-grid-wide">{attentionStocks.map(attentionItem)}</ul>
        )}
      </section>

      <AdUnit placement="bottom" />
    </div>
  );
}
