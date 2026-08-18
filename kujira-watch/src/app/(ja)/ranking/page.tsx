import { Suspense } from "react";
import type { Metadata } from "next";
import Link from "next/link";
import DealTypeBadge from "@/components/DealTypeBadge";
import FilterButtonNav from "@/components/FilterButtonNav";
import ListFallback from "@/components/ListFallback";
import { getFilerWinRates } from "@/lib/investors";
import RankingTabNav from "@/components/RankingTabNav";
import { displayFilerName, formatDate, formatSignedOku } from "@/lib/format";
import { SITE_URL } from "@/lib/site";
import { DEAL_TYPES, type DealType } from "@/types/article";
import AdUnit from "@/components/AdUnit";

export const revalidate = 3600;

// n(結果確定件数)がこれ未満の投資家は表示しない。n=1〜2の勝率は誤差が大きく、
// 「参考にできる」ランキングとして出すには信頼性が低すぎるため。
const MIN_N = 5;

const title = "月間ランキング";
const description =
  "大口投資家のランキング。EDINET大量保有報告書の開示をもとに、" +
  "3ヶ月リターン・買い増し・売却・開示急増の各ランキングをタブで切り替えて見られます。" +
  "このページは買い増し開示の推定取得金額を63営業日(3ヶ月)後までの株価騰落で評価した3ヶ月リターンランキングです。";

// 1ページあたりの表示件数（/stocks・/disclosuresと同値）。
const PER_PAGE = 100;

type Props = {
  searchParams: Promise<{ category?: string; page?: string }>;
};

function buildHref(category: string | null, page: number): string {
  const params = new URLSearchParams();
  if (category) params.set("category", category);
  if (page > 1) params.set("page", String(page));
  const query = params.toString();
  return query ? `/ranking?${query}` : "/ranking";
}

// ページ送りのURLは自分自身をcanonicalにする（1ページ目に集約すると2ページ目以降の
// 内容がインデックスされないため。/investors・/stocksと同じ規律）。
export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { category, page } = await searchParams;
  const pageNumber = Number(page) > 1 ? Number(page) : 1;
  const selected = category && DEAL_TYPES.includes(category as DealType) ? category : null;
  const canonical = `${SITE_URL}${buildHref(selected, pageNumber)}`;
  const suffix = [selected, pageNumber > 1 ? `${pageNumber}ページ目` : null].filter(Boolean).join("・");

  return {
    title: suffix ? `${title}（${suffix}）` : title,
    description,
    alternates: { canonical },
    openGraph: { title, description, url: canonical },
  };
}

// 見出しまでのシェルを先に流し、集計待ちの一覧だけを後から流すためのSuspense境界。
// `searchParams`をここで初めてawaitすることで、ページ本体は待たずに返せる。
export default async function RankingPage({ searchParams }: Props) {
  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: `${SITE_URL}/ranking` },
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
      <h1 className="mb-3 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      <RankingTabNav current="/ranking" />
      <h2 className="mb-1 text-xl font-bold text-brand-navy">3ヶ月リターンランキング</h2>
      <Suspense fallback={<ListFallback rows={12} />}>
        <RankingBody searchParams={searchParams} />
      </Suspense>
      <AdUnit placement="bottom" />
    </div>
  );
}

async function RankingBody({ searchParams }: Props) {
  const { category, page } = await searchParams;
  const filers = await getFilerWinRates(MIN_N);

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
  // 順位はページをまたいで通し番号にする（2ページ目が再び1位から始まらないように）。
  const rankOffset = (currentPage - 1) * PER_PAGE;
  const visibleFilers = matchedFilers.slice(rankOffset, rankOffset + PER_PAGE);

  const updatedAt = filers[0]?.updatedAt;
  const holdDays = filers[0]?.holdDays ?? 63;

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "投資家別 3ヶ月リターンランキング",
    itemListElement: visibleFilers.map((f, index) => ({
      "@type": "ListItem",
      position: rankOffset + index + 1,
      name: f.filerName,
      url: `${SITE_URL}/investors/${encodeURIComponent(f.filerName)}`,
    })),
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
      />
      {updatedAt && (
        <p className="kicker mb-3 text-brand-blue">最終更新: {formatDate(updatedAt)}</p>
      )}
      <p className="mb-4 text-sm text-foreground/50">
        買い増しを開示した投資家を、開示ごとの推定取得金額が{holdDays}営業日（約3ヶ月）後までに
        いくら増減したかの合算で順位づけしています。過去の実績が良い投資家を参考にできます。
        投資助言ではありません。詳しい算出方法・除外条件は
        <Link href="/faq/usage" className="text-brand-blue hover:underline">
          よくある質問
        </Link>
        をご覧ください。
      </p>
      {filers.length > 0 && (
        <FilterButtonNav
          ariaLabel="カテゴリで絞り込む"
          collapsible
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
        <p className="text-foreground/50">該当する投資家がいません。</p>
      ) : (
        // 5列テーブルは狭い画面で投資家名が1文字ずつ折り返されて読めないため、
        // 1件1カードにする（分類は名前の下にバッジで添える）。
        <>
          <p className="kicker mb-2 text-foreground/40">
            順位・投資家 ／ 分類・件数・トータルの推計リターン
          </p>
          <ul className="card-grid card-grid-wide">
            {visibleFilers.map((f, index) => (
              <li key={f.filerName}>
                <Link href={`/investors/${encodeURIComponent(f.filerName)}`} className="card">
                  <span className="flex items-baseline gap-2">
                    <span className="w-5 shrink-0 font-bold tabular-nums text-foreground/40">
                      {rankOffset + index + 1}
                    </span>
                    <span className="min-w-0 grow font-medium text-brand-blue [overflow-wrap:anywhere]">
                      {displayFilerName(f.filerName)}
                    </span>
                  </span>
                  <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 pl-7">
                    <DealTypeBadge dealType={f.category} />
                    <span className="text-xs text-foreground/60">{f.n}件</span>
                    <span
                      className={`ml-auto whitespace-nowrap font-semibold tabular-nums ${
                        f.totalReturnOku >= 0 ? "text-gain" : "text-loss"
                      }`}
                    >
                      {formatSignedOku(f.totalReturnOku)}
                    </span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
          {totalPages > 1 && (
            <nav aria-label="ページ送り" className="mt-6 flex items-center justify-between gap-4 text-sm">
              {currentPage > 1 ? (
                <Link href={buildHref(selectedCategory, currentPage - 1)} className="text-brand-blue hover:underline">
                  ‹ 前の{PER_PAGE}件
                </Link>
              ) : (
                <span />
              )}
              <span className="kicker text-foreground/50">
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
        </>
      )}
    </>
  );
}
