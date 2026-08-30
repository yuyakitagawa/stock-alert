import { Suspense } from "react";
import type { Metadata } from "next";
import Link from "next/link";
import DealTypeIcon from "@/components/DealTypeIcon";
import { siblingDataPages } from "@/lib/nav";
import ListPageNextStep from "@/components/ListPageNextStep";
import DealTypeLabel from "@/components/DealTypeLabel";
import FilterButtonNav from "@/components/FilterButtonNav";
import ListFallback from "@/components/ListFallback";
import RelatedArticles from "@/components/RelatedArticles";
import { getAllFilers, investorPath } from "@/lib/investors";
import {
  getFilerReturnMap,
  formatSignedPercent,
  type FilerReturnBrief,
} from "@/lib/investorReturns";
import { getPublishedFilerNames } from "@/lib/publishedPages";
import { getArticleList } from "@/lib/microcms";
import { displayFilerName, formatDate, latestDateOf, toDateAttr } from "@/lib/format";
import DataUpdatedAt from "@/components/DataUpdatedAt";
import { SITE_URL } from "@/lib/site";
import { DEAL_TYPES, type DealType } from "@/types/article";
import AdUnit from "@/components/AdUnit";

export const revalidate = 3600;

// H1・パンくずは短いラベル（title）のまま、検索結果に出す<title>だけ検索語を入れた形にする。
// GA4の実測（28日）でデータ/一覧ページは940PVのうち889＝95%が内部到達で、入口はわずか51。
// 滞在75秒と全種別で最も長いのに検索から直接来ていない。説明文には既に検索語が入っている
// 一方で<title>が「銘柄ランキング」のような内部呼称のままだったため、そこを揃える（2026-08-27）。
// ※SEOの反映には数日〜数週間かかるので、直後に順位で判定しないこと。
const metaTitle = "大量保有報告書の提出者一覧";
const description =
  "EDINET大量保有報告書（5%ルール）を提出した機関投資家・アクティビストファンド・創業家の資産管理会社などの一覧。投資家別に保有銘柄・保有比率の推移を確認できます。";

// 投資家は約2,900件あり、全件を1ページに並べるとHTMLが1.5MBを超えてTTFBが約1.9秒に
// 悪化していた。1ページあたりの件数を絞り、クロール可能な素のリンクで前後ページを辿れるようにする
// （各投資家ページのURLはサイトマップにも全件載せている）。
const PER_PAGE = 100;

type Props = {
  searchParams: Promise<{ category?: string; page?: string }>;
};

function buildHref(category: string | null, page: number): string {
  const params = new URLSearchParams();
  if (category) params.set("category", category);
  if (page > 1) params.set("page", String(page));
  const query = params.toString();
  return query ? `/investors?${query}` : "/investors";
}

// ページ送りのURLは自分自身をcanonicalにする（1ページ目に集約すると2ページ目以降の
// 内容がインデックスされないため）。
export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { category, page } = await searchParams;
  const pageNumber = Number(page) > 1 ? Number(page) : 1;
  const selected = category && DEAL_TYPES.includes(category as DealType) ? category : null;
  const canonical = `${SITE_URL}${buildHref(selected, pageNumber)}`;
  const suffix = [selected, pageNumber > 1 ? `${pageNumber}ページ目` : null].filter(Boolean).join("・");

  return {
    title: suffix ? `${metaTitle}（${suffix}）` : metaTitle,
    description,
    alternates: { canonical },
    openGraph: { title: metaTitle, description, url: canonical },
  };
}

// 見出しまでのシェルを先に流し、2,900件の集計待ちの一覧だけを後から流すためのSuspense境界。
// `searchParams`をここで初めてawaitすることで、ページ本体は待たずに返せる。
export default async function InvestorsPage({ searchParams }: Props) {
  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "投資家一覧", item: `${SITE_URL}/investors` },
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
        <span className="text-ink-secondary">投資家一覧</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">投資家一覧</h1>
      <Suspense fallback={<ListFallback rows={12} />}>
        <InvestorsBody searchParams={searchParams} />
      </Suspense>
      {/* データページ同士の横移動。ヘッダータブはあるが、GA4実測でTOPへの内部到達398件＝
          他ページからTOPへ戻る動きが多く、横に渡り歩けていなかった（2026-08-27）。 */}
      <ListPageNextStep links={siblingDataPages("/investors")} />
      <AdUnit placement="bottom" />
    </div>
  );
}

async function InvestorsBody({ searchParams }: Props) {
  const [{ category, page }, allFilers, returnByFiler, { contents: latestArticles }, publishedFilers] =
    await Promise.all([
      searchParams,
      getAllFilers(),
      // 「開示のあとどうなったか」＝この投資家が買った銘柄の3ヶ月後リターン。
      // 競合（保有報告のアラートアプリ）は株価を持たないので出せない数字であり、
      // 一覧の段階で見せないと投資家ページまで来た人しか気付けない。
      // 取れなくても一覧自体は成立させる（成績行が消えるだけ）。
      getFilerReturnMap().catch((): Record<string, FilerReturnBrief> => ({})),
      // 一覧の下に添えるアイキャッチ付き記事カード用。取れなくても一覧は成立させる。
      getArticleList({ limit: 4 }).catch(() => ({ contents: [] })),
      getPublishedFilerNames(),
    ]);
  // 解説文が無く開示も1件だけの投資家のページは公開していない（404）ので一覧にも出さない。
  const filers = allFilers.filter((f) => publishedFilers.has(f.filerName));
  // 一覧の更新日は、載っている投資家の最終開示日のうち最も新しいもの。
  const latestDiscDate = latestDateOf(filers.map((f) => f.latestDiscDate));

  const counts = new Map<DealType, number>();
  for (const filer of filers) {
    counts.set(filer.category, (counts.get(filer.category) ?? 0) + 1);
  }
  const activeCategories = DEAL_TYPES.filter((c) => (counts.get(c) ?? 0) > 0);
  const selectedCategory =
    category && DEAL_TYPES.includes(category as DealType) ? (category as DealType) : null;
  const matchedFilers = selectedCategory
    ? filers.filter((f) => f.category === selectedCategory)
    : filers;
  const totalPages = Math.max(1, Math.ceil(matchedFilers.length / PER_PAGE));
  const currentPage = Math.min(Math.max(Number(page) || 1, 1), totalPages);
  const visibleFilers = matchedFilers.slice((currentPage - 1) * PER_PAGE, currentPage * PER_PAGE);

  return (
    <>
      <p className="mb-1 text-sm text-ink-tertiary">
        EDINET大量保有報告書に登場した投資家{filers.length}件。最終開示日が新しい順
        {totalPages > 1 && `（${currentPage}/${totalPages}ページ）`}。
      </p>
      {latestDiscDate && (
        <DataUpdatedAt
          className="mb-4"
          label="最終更新（反映済みの最新開示日）"
          date={latestDiscDate}
          url={`${SITE_URL}/investors`}
        />
      )}
      {filers.length > 0 && (
        <FilterButtonNav
          ariaLabel="カテゴリで絞り込む"
          items={[
            {
              href: buildHref(null, 1),
              label: `すべて（${filers.length}件）`,
              selected: selectedCategory === null,
            },
            ...activeCategories.map((c) => ({
              href: buildHref(c, 1),
              label: `${c}（${counts.get(c)}件）`,
              selected: selectedCategory === c,
            })),
          ]}
        />
      )}
      {visibleFilers.length === 0 ? (
        <p className="text-ink-tertiary">
          {filers.length === 0 ? "投資家データがまだありません。" : "該当する投資家がいません。"}
        </p>
      ) : (
        /* 600件以上を一度に並べるため、カードはMUIコンポーネント（Card/Typography/Chip）ではなく
           素のa.cardにしている。MUI版はHTMLが1.5MBまで膨らみTTFBが約1.9秒だった。
           投資家名が長く2列だと折り返しが増えるため card-grid-wide（2列相当）を使う。 */
        <ul className="card-grid card-grid-wide">
          {visibleFilers.map((filer) => (
            /* 1行目=名前 / 2行目=分類 / 3行目=開示メタ の3行固定。同じ行に流すと
               「創業家の資産管理会社」のような長い分類でだけ折り返しが起きて、
               カードの高さと各行の位置がカードごとにずれていた。 */
            <li key={filer.filerName}>
              <Link
                href={investorPath(filer.filerId, filer.filerName)}
                className="card font-medium text-brand-blue"
              >
                <span className="flex items-start gap-2">
                  {/* 投資家のロゴは持てないため、分類の絵文字＋分類色のアイコンで代替する。 */}
                  <DealTypeIcon dealType={filer.category} />
                  <span className="min-w-0">{displayFilerName(filer.filerName)}</span>
                </span>
                {/* 2・3行目はアイコン分を字下げせずカード左端から。 */}
                <span className="mt-1 block font-normal">
                  <DealTypeLabel dealType={filer.category} />
                </span>
                <span className="block text-xs font-normal text-ink-tertiary">
                  保有開示{filer.holdingCount}件・最終開示
                  <time dateTime={toDateAttr(filer.latestDiscDate)}>
                    {formatDate(filer.latestDiscDate)}
                  </time>
                </span>
                {/* 4行目。買い開示が少ない投資家はビューに載らないので行ごと出ない
                    （高さがカードごとにずれるが、成績のある投資家を目立たせる方を採る）。 */}
                {(() => {
                  const record = returnByFiler[filer.filerName];
                  if (!record) return null;
                  return (
                    <span className="mt-1 block text-xs font-normal">
                      <span className="text-ink-tertiary">開示3ヶ月後 </span>
                      <span
                        className={
                          record.avgReturn >= 0
                            ? "font-medium text-brand-blue"
                            : "font-medium text-red-600"
                        }
                      >
                        平均{formatSignedPercent(record.avgReturn)}
                      </span>
                      <span className="text-ink-tertiary">
                        ・勝率{record.winRate}%（{record.positionCount}件）
                      </span>
                    </span>
                  );
                })()}
              </Link>
            </li>
          ))}
        </ul>
      )}
      {totalPages > 1 && (
        <nav aria-label="ページ送り" className="mt-6 flex items-center justify-between gap-4 text-sm">
          {currentPage > 1 ? (
            <Link href={buildHref(selectedCategory, currentPage - 1)} className="text-brand-blue hover:underline">
              ‹ 前の{PER_PAGE}件
            </Link>
          ) : (
            <span />
          )}
          <span className="kicker text-ink-tertiary">
            {currentPage} / {totalPages}
          </span>
          {currentPage < totalPages ? (
            <Link href={buildHref(selectedCategory, currentPage + 1)} className="text-brand-blue hover:underline">
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
          lead="投資家一覧に載っている大口投資家たちの、直近の取引の解説記事です。"
          articles={latestArticles}
        />
      </div>
    </>
  );
}
