import type { Metadata } from "next";
import Link from "next/link";
import InfoTip from "@/components/InfoTip";
import { getHoldingAmountsInRange, getHoldingsInRange } from "@/lib/investors";
import { getAllListedCodes, getCompanyBriefs } from "@/lib/companyInfo";
import RelatedArticles from "@/components/RelatedArticles";
import TrendingDirectionTable from "@/components/TrendingDirectionTable";
import { getArticleList } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { buildTrendingIssuers, selectDirection } from "@/lib/trendingStats";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

// 比較期間。週次(7日)だと開示の少ない週に振り回され、暦月だと月初は「数日 vs 1か月」の
// 比較になってしまうため、常に同じ長さで比べられる30日固定にする。
const WINDOW_DAYS = 30;

// 構造化データに載せる件数。オートページャーで追加描画される分はクライアント側の
// 処理なので、クロール時点でサーバーが返すHTML（TrendingTableのINITIAL_COUNT）に合わせる。
const JSON_LD_COUNT = 30;

const url = `${SITE_URL}/trending`;
const title = "銘柄ランキング";

export const metadata: Metadata = {
  title,
  description: `直近${WINDOW_DAYS}日間で大量保有報告書の開示が増えた銘柄を、その前の${WINDOW_DAYS}日間と比べた増加件数順にランキング。開示の件数と推定売買金額を並べて表示し、買い・売り・両方で絞り込めます。EDINETの開示データをもとに毎日更新しています。`,
  alternates: { canonical: url },
  openGraph: { title, url },
};

function daysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

export default async function TrendingPage() {
  const currentFrom = daysAgo(WINDOW_DAYS - 1);
  const rangeFrom = daysAgo(WINDOW_DAYS * 2 - 1);
  const rangeTo = daysAgo(0);

  const [rows, amountByDocId, listedCodes, { contents: recentArticles }] = await Promise.all([
    getHoldingsInRange(rangeFrom, rangeTo),
    // 金額は件数に添えるだけの情報なので、取れなくてもランキング自体は成立させる。
    getHoldingAmountsInRange(rangeFrom, rangeTo).catch(() => ({})),
    getAllListedCodes().catch(() => new Set<string>()),
    // ランキング銘柄の解説記事（アイキャッチ付きカード）用。取れなくてもページは成立させる。
    getArticleList({ limit: 30 }).catch(() => ({ contents: [] })),
  ]);

  // 件数制限なし。直近30日で開示が増えた銘柄をすべて出す（買い・売り・両方のいずれかで増えたもの）。
  const trendingIssuers = buildTrendingIssuers(rows, currentFrom, amountByDocId);

  // 社名と証券コードだけでは何の会社か分からないため、事業内容を1行添える。
  // 事業内容(jpx_stock_list.description)は未生成の銘柄があるので、その場合は業種で代替する。
  const briefs = await getCompanyBriefs(trendingIssuers.map((entry) => entry.key));
  const noteOf = (code: string): string | null => {
    const brief = briefs.get(code);
    return brief?.description ?? brief?.sector ?? null;
  };

  // 銘柄ページ(/stocks/[code])は上場銘柄マスターに載っていれば解説記事が無くても
  // 開示履歴＋会社情報で成立する。マスターに無いコード（上場廃止等）だけ
  // リンクにせずテキストのまま出す（404へのリンクを作らない）。

  // href・noteはサーバー側で解決してから渡す（TrendingDirectionTableはクライアント
  // コンポーネントなので関数propsを境界を越えて渡せない）。
  const trendingItems = trendingIssuers.map((entry) => ({
    ...entry,
    href: listedCodes.has(entry.key) ? `/stocks/${entry.key}` : null,
    note: noteOf(entry.key),
    sector: briefs.get(entry.key)?.sector ?? null,
  }));

  // ランキング入りした銘柄の解説記事（新着順・1銘柄1件まで）。表だけでは取引の中身が
  // 分からないため、アイキャッチ付きカードで記事への導線を添える。
  const trendingCodes = new Set(trendingIssuers.map((entry) => entry.key));
  const seenCodes = new Set<string>();
  const trendingArticles = recentArticles.filter((article) => {
    if (!trendingCodes.has(article.stockCode) || seenCodes.has(article.stockCode)) return false;
    seenCodes.add(article.stockCode);
    return true;
  }).slice(0, 4);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  // 可視コンテンツと合わせ、実際にリンクしている銘柄のみItemList化する。
  // 絞り込みの初期値は「買い」なので、SSRされるHTMLと同じく買いの一覧を載せる。
  const defaultViewItems = selectDirection(trendingItems, "buy");
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `買いの大量保有報告書の開示が増えた銘柄（直近${WINDOW_DAYS}日）`,
    itemListElement: defaultViewItems
      .slice(0, JSON_LD_COUNT)
      .filter((entry) => entry.href !== null)
      .map((entry, index) => ({
        "@type": "ListItem",
        position: index + 1,
        name: entry.label,
        url: `${SITE_URL}/stocks/${entry.key}`,
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
          直近{WINDOW_DAYS}日間で大口投資家の取引（大量保有報告書）が増えた銘柄のランキングです。
          <InfoTip
            content={
              <>
                前の{WINDOW_DAYS}日間と比べた増加件数順に並べています。保有比率が増えた「買い」（初期表示）・減った「売り」・その両方で絞り込めます。
                各銘柄の推定売買金額は「保有比率の変化幅×発行済株式数×開示日の終値」の概算です。
              </>
            }
          />
          <br />
          投資家ごとの成績は
          <Link href="/ranking/returns" className="text-brand-blue hover:underline">
            3ヶ月リターンランキング
          </Link>
          、見方は
          <Link href="/faq/usage" className="text-brand-blue hover:underline">
            よくある質問
          </Link>
          へ。
        </p>
      </div>

      <section className="mb-10">
        <TrendingDirectionTable items={trendingItems} windowDays={WINDOW_DAYS} />
        <p className="mt-3 text-xs leading-relaxed text-foreground/40">
          ※金額はEDINET開示に取引金額の記載が無いための概算です。売買を伴わない訂正報告書や、
          株価・発行済株式数が取れない銘柄は金額に含めていない（件数には含む）ため、件数と金額は必ずしも比例しません。
        </p>
      </section>

      <RelatedArticles
        title="ランキング銘柄の解説記事"
        lead="ランキングに入った銘柄の直近の大口取引を、取引ごとの解説記事で読めます。"
        articles={trendingArticles}
      />

      <AdUnit placement="bottom" />
    </div>
  );
}
