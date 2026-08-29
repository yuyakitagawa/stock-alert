import type { Metadata } from "next";
import Link from "next/link";
import InfoTip from "@/components/InfoTip";
import { siblingDataPages } from "@/lib/nav";
import ListPageNextStep from "@/components/ListPageNextStep";
import {
  getActivistHoldingsSummary,
  getActivistRecentMoves,
  type ActivistMove,
} from "@/lib/activists";
import { displayFilerName, formatDate } from "@/lib/format";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import RatioTransition from "@/components/RatioTransition";
import RelatedArticles from "@/components/RelatedArticles";
import SectorIcon from "@/components/SectorIcon";
import MagnitudeBar from "@/components/MagnitudeBar";
import AdUnit from "@/components/AdUnit";
import { getCompanyBriefs, type CompanyBrief } from "@/lib/companyInfo";
import { getPublishedFilerNames, getPublishedStockCodes } from "@/lib/publishedPages";
import { getFilerIdMap, investorPath } from "@/lib/investors";

export const revalidate = 3600;

// 買い入れの集計期間（/trendingと同じ7日＝直近1週間。30日窓は毎日ほぼ同じ顔ぶれが並び
// 「直近の動き」が見えなくなるため2026-08-27に短縮した。7日でも買い入れは20銘柄前後出る）。
const MOVES_WINDOW_DAYS = 7;
// アクティビスト注目銘柄の表示件数。全件を一度に並べるが、開示が集中した週に全部描画すると
// ページのHTMLが重くなるため件数自体をここで打ち切る
// （打ち切った件数は画面にも明記して、全件があるように見せない）。
const ATTENTION_RENDER_LIMIT = 100;

const url = `${SITE_URL}/activists`;
const title = "アクティビスト注目銘柄";

export const metadata: Metadata = {
  title,
  description:
    "アクティビスト（物言う株主）が直近7日に大きく買い入れた注目銘柄を、EDINET大量保有報告書をもとに増加幅の大きい順で一覧。保有比率・開示日つきで毎日更新しています。",
  alternates: { canonical: url },
  openGraph: { title, url },
};

export default async function ActivistsPage() {
  const [summary, recentMoves, publishedCodes, filerIds, { contents: activistArticles }, publishedFilers] = await Promise.all([
    getActivistHoldingsSummary(),
    getActivistRecentMoves(MOVES_WINDOW_DAYS).catch(() => []),
    getPublishedStockCodes().catch(() => new Set<string>()),
    getFilerIdMap().catch(() => ({}) as Record<string, number>),
    // アイキャッチ付き記事カード用のアクティビスト分類の最新記事。取れなくてもページは成立させる。
    getArticleList({ dealType: "アクティビスト", limit: 8 }).catch(() => ({ contents: [] })),
    getPublishedFilerNames().catch(() => new Set<string>()),
  ]);

  // 銘柄ページ(/stocks/[code])は解説記事か事業内容の説明があるものだけ公開している。
  // 公開していないコードはリンクにせずテキストのまま出す（404へのリンクを作らない。
  // /trendingと同じ規律。判定は lib/publishedPages.ts）。

  const stockLabel = (issuerName: string, issuerCode: string) =>
    publishedCodes.has(issuerCode) ? (
      <Link href={`/stocks/${issuerCode}`} className="text-brand-blue hover:underline">
        {issuerName}（{issuerCode}）
      </Link>
    ) : (
      <span className="text-foreground/80">
        {issuerName}（{issuerCode}）
      </span>
    );

  // 注目銘柄=直近7日にアクティビストが保有比率を増やした（買い入れた）銘柄。
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
  // 買い入れ幅の量感バーの基準（表示分の最大増加幅）。
  const maxDelta = attentionStocks.reduce((max, row) => Math.max(max, row.totalDelta), 0);

  // 銘柄カードの業種アイコン用（会社ロゴは持てないため業種で代替）。表示分だけ一括取得。
  const briefs = await getCompanyBriefs(attentionStocks.map((row) => row.issuerCode)).catch(
    () => new Map<string, CompanyBrief>()
  );

  const attentionItem = (row: AttentionRow) => (
    <li key={row.issuerCode} className="card text-sm">
      {/* 1行目=アイコン＋銘柄名、2行目以降=明細をカード左端から（アイコン分の字下げはしない）。 */}
      <div className="flex items-start gap-2">
        <SectorIcon sector={briefs.get(row.issuerCode)?.sector} size="lg" />
        <div className="min-w-0">
          <span className="font-medium">{stockLabel(row.issuerName, row.issuerCode)}</span>
          <span className="ml-2 whitespace-nowrap text-xs font-bold text-gain">▲{row.totalDelta}pt買い入れ</span>
          {row.multiHolder && (
            <span className="kicker ml-2 whitespace-nowrap text-brand-gold">複数ファンド保有</span>
          )}
        </div>
      </div>
      {/* 買い入れ幅の量感バー。1位だけ金色にして先頭が読み取れるようにする。 */}
      <MagnitudeBar
        value={row.totalDelta}
        max={maxDelta}
        tone={row.totalDelta === maxDelta ? "gold" : "gain"}
      />
      <ul className="mt-1 space-y-0.5 text-xs text-foreground/60">
        {row.buys.map((move) => (
          <li key={move.docId}>
            {publishedFilers.has(move.filerName) ? (
              <Link
                href={investorPath(filerIds[move.filerName], move.filerName)}
                className="text-brand-blue hover:underline"
              >
                {displayFilerName(move.filerName)}
              </Link>
            ) : (
              <span className="text-foreground/80">{displayFilerName(move.filerName)}</span>
            )}
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
          アクティビスト（物言う株主）が直近{MOVES_WINDOW_DAYS}日に買い入れた銘柄を、増加幅の大きい順に表示しています。
          <InfoTip
            content={
              <>
                {SITE_NAME}が投資家分類「アクティビスト」と判定した提出者が保有比率を増やした銘柄を、増加幅の合計順に並べています。出典はEDINET大量保有報告書。詳しくは
                <Link href="/faq/usage" className="underline">よくある質問</Link>
                をご覧ください。
              </>
            }
          />
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

      <RelatedArticles
        title="アクティビストの解説記事"
        lead="アクティビストによる直近の取引を、取引ごとの解説記事で読めます。"
        articles={activistArticles}
      />

      {/* データページ同士の横移動。ヘッダータブはあるが、GA4実測でTOPへの内部到達398件＝
          他ページからTOPへ戻る動きが多く、横に渡り歩けていなかった（2026-08-27）。 */}
      <ListPageNextStep links={siblingDataPages("/activists")} />
      <AdUnit placement="bottom" />
    </div>
  );
}
