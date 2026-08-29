import type { Metadata } from "next";
import Link from "next/link";
import InfoTip from "@/components/InfoTip";
import { siblingDataPages } from "@/lib/nav";
import ListPageNextStep from "@/components/ListPageNextStep";
import { getHoldingAmountsInRange, getHoldingsInRange } from "@/lib/investors";
import { getCompanyBriefs } from "@/lib/companyInfo";
import { getPublishedStockCodes } from "@/lib/publishedPages";
import RelatedArticles from "@/components/RelatedArticles";
import TrendingDirectionTable from "@/components/TrendingDirectionTable";
import { getArticleList } from "@/lib/microcms";
import { SITE_URL } from "@/lib/site";
import { buildTrendingIssuers, selectDirection } from "@/lib/trendingStats";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

// 比較期間。「今どの銘柄に大口の資金が入っているか」を今週の動きとして見せるため、
// 直近7日 vs その前の7日で比べる（暦週ではなく常に同じ長さの7日固定にして、
// 月初・週初でも「数日 vs 1週間」の歪んだ比較にならないようにする）。
const WINDOW_DAYS = 7;

// 構造化データに載せる件数。オートページャーで追加描画される分はクライアント側の
// 処理なので、クロール時点でサーバーが返すHTML（TrendingTableのINITIAL_COUNT）に合わせる。
const JSON_LD_COUNT = 30;

// 一覧の下に添えるアイキャッチ付き記事カードの件数。先頭1件は2列ぶんの幅で出る
// （`RelatedArticles`）。データページは画像が最下部にしか無く見た目が弱かったため
// 2026-08-29に4→8へ増やした。8件を埋めるには「ランキング入りした銘柄の記事」で
// 絞り込んだ後に8件残る必要があるので、取得元の新着記事も30→60件に広げている。
const RELATED_ARTICLE_SIZE = 8;

const url = `${SITE_URL}/trending`;
const title = "銘柄ランキング";
// H1・パンくずは短いラベル（title）のまま、検索結果に出す<title>だけ検索語を入れた形にする。
// GA4の実測（28日）でデータ/一覧ページは940PVのうち889＝95%が内部到達で、入口はわずか51。
// 滞在75秒と全種別で最も長いのに検索から直接来ていない。説明文には既に検索語が入っている
// 一方で<title>が「銘柄ランキング」のような内部呼称のままだったため、そこを揃える（2026-08-27）。
// ※SEOの反映には数日〜数週間かかるので、直後に順位で判定しないこと。
const metaTitle = "大量保有報告書が増えた銘柄ランキング";

export const metadata: Metadata = {
  title: metaTitle,
  description: `直近${WINDOW_DAYS}日間で大量保有報告書の開示が増えた銘柄を、推定売買金額の大きい順にランキング。開示の件数と金額を並べて表示し、買い・売り・両方で絞り込めます。EDINETの開示データをもとに毎日更新しています。`,
  alternates: { canonical: url },
  openGraph: { title: metaTitle, url },
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

  const [rows, amountByDocId, publishedCodes, { contents: recentArticles }] = await Promise.all([
    getHoldingsInRange(rangeFrom, rangeTo),
    // 金額はランキングの並べ替え軸だが、取れなかった場合も件数（増加件数順）にフォールバックして
    // ページ自体は成立させる。金額が読めないことを理由に一覧を落とすほうが読者の損失が大きい。
    getHoldingAmountsInRange(rangeFrom, rangeTo).catch(() => ({})),
    getPublishedStockCodes().catch(() => new Set<string>()),
    // ランキング銘柄の解説記事（アイキャッチ付きカード）用。取れなくてもページは成立させる。
    getArticleList({ limit: 60 }).catch(() => ({ contents: [] })),
  ]);

  // 件数制限なし。直近7日で開示が増えた銘柄をすべて出す（買い・売り・両方のいずれかで増えたもの）。
  const trendingIssuers = buildTrendingIssuers(rows, currentFrom, amountByDocId);

  // 社名と証券コードだけでは何の会社か分からないため、事業内容を1行添える。
  // 事業内容(jpx_stock_list.description)は未生成の銘柄があるので、その場合は業種で代替する。
  const briefs = await getCompanyBriefs(trendingIssuers.map((entry) => entry.key));
  const noteOf = (code: string): string | null => {
    const brief = briefs.get(code);
    return brief?.description ?? brief?.sector ?? null;
  };

  // 銘柄ページ(/stocks/[code])は解説記事か事業内容の説明があるものだけ公開している。
  // 公開していないコードはリンクにせずテキストのまま出す（404へのリンクを作らない。
  // 判定は lib/publishedPages.ts でページ側・サイトマップ側と共通）。

  // href・noteはサーバー側で解決してから渡す（TrendingDirectionTableはクライアント
  // コンポーネントなので関数propsを境界を越えて渡せない）。
  const trendingItems = trendingIssuers.map((entry) => ({
    ...entry,
    href: publishedCodes.has(entry.key) ? `/stocks/${entry.key}` : null,
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
  }).slice(0, RELATED_ARTICLE_SIZE);

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
    name: `買いの大量保有報告書の開示が増えた銘柄の推定売買金額ランキング（直近${WINDOW_DAYS}日）`,
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
前の{WINDOW_DAYS}日間より開示が増えた銘柄を、推定売買金額の大きい順に並べています（金額が同じなら増加件数順）。保有比率が増えた「買い」（初期表示）・減った「売り」・その両方で絞り込めます。
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
          株価・発行済株式数が取れない銘柄は金額を推定できないため「金額不明」として一覧の下側に並びます
          （件数には含みます）。
        </p>
      </section>

      <RelatedArticles
        title="ランキング銘柄の解説記事"
        lead="ランキングに入った銘柄の直近の大口取引を、取引ごとの解説記事で読めます。"
        articles={trendingArticles}
      />

      {/* データページ同士の横移動。ヘッダータブはあるが、GA4実測でTOPへの内部到達398件＝
          他ページからTOPへ戻る動きが多く、横に渡り歩けていなかった（2026-08-27）。 */}
      <ListPageNextStep links={siblingDataPages("/trending")} />
      <AdUnit placement="bottom" />
    </div>
  );
}
