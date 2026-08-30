import { Suspense } from "react";
import type { Metadata } from "next";
import Link from "next/link";
import FilterButtonNav from "@/components/FilterButtonNav";
import { siblingDataPages } from "@/lib/nav";
import ListPageNextStep from "@/components/ListPageNextStep";
import ListFallback from "@/components/ListFallback";
import RelatedArticles from "@/components/RelatedArticles";
import SectorIcon from "@/components/SectorIcon";
import { getAllStocksForIndex, getArticleList } from "@/lib/microcms";
import { getAllSectorsByCode } from "@/lib/companyInfo";
import { getPublishedStockCodes } from "@/lib/publishedPages";
import { formatDate } from "@/lib/format";
import { SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

export const revalidate = 300;

// H1・パンくずは短いラベル（title）のまま、検索結果に出す<title>だけ検索語を入れた形にする。
// GA4の実測（28日）でデータ/一覧ページは940PVのうち889＝95%が内部到達で、入口はわずか51。
// 滞在75秒と全種別で最も長いのに検索から直接来ていない。説明文には既に検索語が入っている
// 一方で<title>が「銘柄ランキング」のような内部呼称のままだったため、そこを揃える（2026-08-27）。
// ※SEOの反映には数日〜数週間かかるので、直後に順位で判定しないこと。
const metaTitle = "大量保有報告書が出た銘柄一覧";
const description =
  "EDINET大量保有報告書（5%ルール）・自社株買いなど、大口投資家の動きが開示された銘柄の一覧。銘柄別に保有・取引の履歴を確認できます。";

// 1ページあたりの表示件数。3列グリッドで約33行になり、ページ送りせずに
// 端まで見渡せる上限としてこの値にしている。
const PER_PAGE = 100;

type Props = {
  searchParams: Promise<{ sector?: string; page?: string }>;
};

function buildHref(sector: string | null, page: number): string {
  const params = new URLSearchParams();
  if (sector) params.set("sector", sector);
  if (page > 1) params.set("page", String(page));
  const query = params.toString();
  return query ? `/stocks?${query}` : "/stocks";
}

// ページ送りのURLは自分自身をcanonicalにする（1ページ目に集約すると2ページ目以降の
// 内容がインデックスされないため。/investorsと同じ規律）。
export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { sector, page } = await searchParams;
  const pageNumber = Number(page) > 1 ? Number(page) : 1;
  const selected = sector ?? null;
  const canonical = `${SITE_URL}${buildHref(selected, pageNumber)}`;
  const suffix = [selected, pageNumber > 1 ? `${pageNumber}ページ目` : null].filter(Boolean).join("・");

  return {
    title: suffix ? `${metaTitle}（${suffix}）` : metaTitle,
    description,
    alternates: { canonical },
    openGraph: { title: metaTitle, description, url: canonical },
  };
}

// 見出しまでのシェルを先に流し、銘柄一覧の取得待ちだけを後から流すためのSuspense境界。
// `searchParams`をここで初めてawaitすることで、ページ本体は待たずに返せる。
export default async function StocksIndexPage({ searchParams }: Props) {
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
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-ink-secondary">銘柄一覧</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">銘柄一覧</h1>
      <Suspense fallback={<ListFallback rows={12} />}>
        <StocksBody searchParams={searchParams} />
      </Suspense>
      {/* データページ同士の横移動。ヘッダータブはあるが、GA4実測でTOPへの内部到達398件＝
          他ページからTOPへ戻る動きが多く、横に渡り歩けていなかった（2026-08-27）。 */}
      <ListPageNextStep links={siblingDataPages("/stocks")} />
      <AdUnit placement="bottom" />
    </div>
  );
}

async function StocksBody({ searchParams }: Props) {
  const [{ sector, page }, allStocks, sectorByCode, { contents: latestArticles }, publishedCodes] =
    await Promise.all([
      searchParams,
      getAllStocksForIndex(),
      getAllSectorsByCode(),
      // 一覧の下に添えるアイキャッチ付き記事カード用。取れなくても一覧は成立させる。
      getArticleList({ limit: 4 }).catch(() => ({ contents: [] })),
      getPublishedStockCodes(),
    ]);
  // 解説記事も事業内容の説明も無い銘柄のページは公開していない（404）ので一覧にも出さない。
  const stocks = allStocks.filter((s) => publishedCodes.has(s.stockCode));

  const counts = new Map<string, number>();
  for (const s of stocks) {
    const sec = sectorByCode.get(s.stockCode);
    if (sec) counts.set(sec, (counts.get(sec) ?? 0) + 1);
  }
  const sectors = Array.from(counts.keys()).sort((a, b) => a.localeCompare(b, "ja"));
  const selectedSector = sector && counts.has(sector) ? sector : null;
  const matchedStocks = selectedSector
    ? stocks.filter((s) => sectorByCode.get(s.stockCode) === selectedSector)
    : stocks;
  const totalPages = Math.max(1, Math.ceil(matchedStocks.length / PER_PAGE));
  const currentPage = Math.min(Math.max(Number(page) || 1, 1), totalPages);
  const visibleStocks = matchedStocks.slice((currentPage - 1) * PER_PAGE, currentPage * PER_PAGE);

  return (
    <>
      <p className="mb-4 text-sm text-ink-tertiary">
        大量保有・自社株買いの開示があった銘柄{stocks.length}件。証券コード順
        {totalPages > 1 && `（${currentPage}/${totalPages}ページ）`}。
      </p>
      {sectors.length > 0 && (
        <FilterButtonNav
          ariaLabel="業種で絞り込む"
          items={[
            {
              href: buildHref(null, 1),
              label: `すべて（${stocks.length}件）`,
              selected: selectedSector === null,
            },
            ...sectors.map((sec) => ({
              href: buildHref(sec, 1),
              label: `${sec}（${counts.get(sec)}件）`,
              selected: selectedSector === sec,
            })),
          ]}
        />
      )}
      {visibleStocks.length === 0 ? (
        <p className="text-ink-tertiary">
          {stocks.length === 0 ? "銘柄データがまだありません。" : "該当する銘柄がありません。"}
        </p>
      ) : (
        /* 全銘柄（数百件）を並べるため、カードはグリッドで多列化する。
           1列のままカード化すると縦の総量が行リストの約3倍になり索引として使えない。 */
        <ul className="card-grid">
          {visibleStocks.map((stock) => (
            <li key={stock.stockCode}>
              <Link href={`/stocks/${stock.stockCode}`} className="card">
                <span className="flex items-start gap-2">
                  <SectorIcon sector={sectorByCode.get(stock.stockCode)} />
                  <span className="min-w-0 font-medium text-brand-blue">
                    {stock.stockName}（{stock.stockCode}）
                  </span>
                </span>
                {/* 2・3行目はアイコン分を字下げせずカード左端から。 */}
                <span className="mt-1 block text-xs text-ink-tertiary">
                  {sectorByCode.get(stock.stockCode) && `${sectorByCode.get(stock.stockCode)}・`}
                  記事{stock.articleCount}件
                </span>
                <span className="block text-xs text-ink-tertiary">
                  最終開示{formatDate(stock.latestDealDate)}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
      {totalPages > 1 && (
        <nav aria-label="ページ送り" className="mt-6 flex items-center justify-between gap-4 text-sm">
          {currentPage > 1 ? (
            <Link href={buildHref(selectedSector, currentPage - 1)} className="text-brand-blue hover:underline">
              ‹ 前の{PER_PAGE}件
            </Link>
          ) : (
            <span />
          )}
          <span className="kicker text-ink-tertiary">
            {currentPage} / {totalPages}
          </span>
          {currentPage < totalPages ? (
            <Link href={buildHref(selectedSector, currentPage + 1)} className="text-brand-blue hover:underline">
              次の{PER_PAGE}件 ›
            </Link>
          ) : (
            <span />
          )}
        </nav>
      )}
      <div className="mt-10">
        <RelatedArticles
          title="最新の解説記事"
          lead="一覧の銘柄で直近にあった大口取引の解説記事です。"
          articles={latestArticles}
        />
      </div>
    </>
  );
}
