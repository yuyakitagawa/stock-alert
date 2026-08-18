import { Suspense } from "react";
import type { Metadata } from "next";
import Link from "next/link";
import FilterButtonNav from "@/components/FilterButtonNav";
import ListFallback from "@/components/ListFallback";
import RatioTransition from "@/components/RatioTransition";
import { formatDate } from "@/lib/format";
import { getAllStocksForIndex } from "@/lib/microcms";
import {
  DISCLOSURES_PER_PAGE,
  disclosureKindLabel,
  edinetPdfUrl,
  getDisclosureCounts,
  getDisclosuresPage,
  type DisclosureRow,
  type DisclosureTypeFilter,
} from "@/lib/disclosures";
import { SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

const title = "大量保有報告書の開示速報";
const description =
  "EDINETに提出された大量保有報告書・変更報告書の全件を提出日ごとに一覧。保有比率の増減、新規・変更・訂正の種別、原文PDFへのリンクつきで毎日更新しています。";

type Props = {
  searchParams: Promise<{ type?: string; page?: string }>;
};

const TYPE_FILTERS: { key: DisclosureTypeFilter; label: string }[] = [
  { key: "new", label: "新規（大量保有報告書）" },
  { key: "change", label: "変更報告書" },
  { key: "correction", label: "訂正報告書" },
];

function parseTypeFilter(type?: string): DisclosureTypeFilter | null {
  return type === "new" || type === "change" || type === "correction" ? type : null;
}

function buildHref(type: DisclosureTypeFilter | null, page: number): string {
  const params = new URLSearchParams();
  if (type) params.set("type", type);
  if (page > 1) params.set("page", String(page));
  const query = params.toString();
  return query ? `/disclosures?${query}` : "/disclosures";
}

// ページ送り・フィルターのURLは自分自身をcanonicalにする（/investorsと同じ方針）。
export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { type, page } = await searchParams;
  const selectedType = parseTypeFilter(type);
  const pageNumber = Number(page) > 1 ? Number(page) : 1;
  const canonical = `${SITE_URL}${buildHref(selectedType, pageNumber)}`;
  const typeLabel = TYPE_FILTERS.find((t) => t.key === selectedType)?.label ?? null;
  const suffix = [typeLabel, pageNumber > 1 ? `${pageNumber}ページ目` : null]
    .filter(Boolean)
    .join("・");

  return {
    title: suffix ? `${title}（${suffix}）` : title,
    description,
    alternates: { canonical },
    openGraph: { title, description, url: canonical },
  };
}

// 提出時刻（"YYYY-MM-DD HH:MM"）から時刻部分だけを取り出す。形式が違う行は出さない。
function submitTime(submitDate: string): string | null {
  const match = submitDate.match(/\d{2}:\d{2}$/);
  return match ? match[0] : null;
}

function groupByDiscDate(rows: DisclosureRow[]): { discDate: string; rows: DisclosureRow[] }[] {
  const groups: { discDate: string; rows: DisclosureRow[] }[] = [];
  for (const row of rows) {
    const last = groups[groups.length - 1];
    if (last && last.discDate === row.discDate) {
      last.rows.push(row);
    } else {
      groups.push({ discDate: row.discDate, rows: [row] });
    }
  }
  return groups;
}

// 見出しまでのシェルを先に流し、開示の件数集計とページ取得を待つ部分だけを後から流す。
// `searchParams`をここで初めてawaitすることで、ページ本体は待たずに返せる。
export default async function DisclosuresPage({ searchParams }: Props) {
  return (
    <div>
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">{title}</span>
      </nav>

      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      <Suspense fallback={<ListFallback rows={12} />}>
        <DisclosuresBody searchParams={searchParams} />
      </Suspense>

      <AdUnit placement="bottom" />
    </div>
  );
}

async function DisclosuresBody({ searchParams }: Props) {
  const { type, page } = await searchParams;
  const selectedType = parseTypeFilter(type);

  const counts = await getDisclosureCounts();
  const totalCount = selectedType ? counts[selectedType] : counts.all;
  const totalPages = Math.max(1, Math.ceil(totalCount / DISCLOSURES_PER_PAGE));
  const currentPage = Math.min(Math.max(Number(page) || 1, 1), totalPages);

  // 銘柄リンクの解決に失敗しても開示一覧自体は表示する（リンク無しに退化するだけ）。
  const [rows, stocksWithArticles] = await Promise.all([
    getDisclosuresPage(currentPage, selectedType),
    getAllStocksForIndex().catch(() => []),
  ]);

  // 銘柄ページ(/stocks/[code])は記事がある銘柄にしか存在しないため、
  // 記事の無い銘柄はリンクにせずテキストのまま出す（/trendingと同じ規律）。
  const codesWithArticles = new Set(stocksWithArticles.map((s) => s.stockCode));

  const groups = groupByDiscDate(rows);
  const url = `${SITE_URL}${buildHref(selectedType, currentPage)}`;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <p className="mb-4 text-sm leading-relaxed text-foreground/70">
        EDINETに提出された取引を提出日の新しい順に一覧しています。
        {totalPages > 1 && `（${currentPage}/${totalPages}ページ）`}
        <br />
        ※開示は最大5営業日遅れて提出される点にご注意ください。詳しくは
        <Link href="/faq/usage" className="text-brand-blue hover:underline">
          よくある質問
        </Link>
        をご覧ください。
      </p>

      <FilterButtonNav
        ariaLabel="報告書の種別で絞り込む"
        items={[
          {
            href: buildHref(null, 1),
            label: `すべて（${counts.all.toLocaleString()}件）`,
            selected: selectedType === null,
          },
          {
            href: buildHref("new", 1),
            label: `新規（${counts.new.toLocaleString()}件）`,
            selected: selectedType === "new",
          },
          {
            href: buildHref("change", 1),
            label: `変更（${counts.change.toLocaleString()}件）`,
            selected: selectedType === "change",
          },
          {
            href: buildHref("correction", 1),
            label: `訂正（${counts.correction.toLocaleString()}件）`,
            selected: selectedType === "correction",
          },
        ]}
      />

      {rows.length === 0 ? (
        <p className="text-foreground/50">開示データがまだありません。</p>
      ) : (
        groups.map((group) => (
          <section key={group.discDate} className="mb-8">
            <h2 className="mb-1 border-b border-rule pb-1 text-lg font-bold text-brand-navy">
              {formatDate(group.discDate)}提出
              <span className="ml-2 text-xs font-normal text-foreground/50">
                {group.rows.length}件
              </span>
            </h2>
            {/* 1ページ100件を並べるため、行はMUIではなく素のul/li（/investorsの教訓）。 */}
            <ul>
              {group.rows.map((row) => {
                const kind = disclosureKindLabel(row);
                const stockHref = codesWithArticles.has(row.issuerCode)
                  ? `/stocks/${row.issuerCode}`
                  : null;
                const time = submitTime(row.submitDate);
                return (
                  <li
                    key={row.docId}
                    className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-rule py-2.5 text-sm"
                  >
                    <span
                      className={`kicker whitespace-nowrap ${
                        kind === "新規"
                          ? "text-brand-gold"
                          : kind === "訂正"
                            ? "text-foreground/40"
                            : "text-foreground/50"
                      }`}
                    >
                      {kind}
                    </span>
                    <span className="font-medium">
                      {stockHref ? (
                        <Link href={stockHref} className="text-brand-blue hover:underline">
                          {row.issuerName}（{row.issuerCode}）
                        </Link>
                      ) : (
                        <span className="text-foreground/80">
                          {row.issuerName}（{row.issuerCode}）
                        </span>
                      )}
                    </span>
                    <RatioTransition ratio={row.holdingRatio} prior={row.holdingRatioPrior} />
                    <span className="text-foreground/60">
                      提出者:{" "}
                      <Link
                        href={`/investors/${encodeURIComponent(row.filerName)}`}
                        className="text-brand-blue hover:underline"
                      >
                        {row.filerName}
                      </Link>
                    </span>
                    <span className="ml-auto flex items-baseline gap-x-3 text-xs text-foreground/50">
                      {time && <span>{time}</span>}
                      <a
                        href={edinetPdfUrl(row.docId)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="whitespace-nowrap text-brand-blue hover:underline"
                      >
                        原文PDF ↗
                      </a>
                    </span>
                  </li>
                );
              })}
            </ul>
          </section>
        ))
      )}

      {totalPages > 1 && (
        <nav aria-label="ページ送り" className="mt-6 flex items-center justify-between gap-4 text-sm">
          {currentPage > 1 ? (
            <Link
              href={buildHref(selectedType, currentPage - 1)}
              className="text-brand-blue hover:underline"
            >
              ‹ 新しい{DISCLOSURES_PER_PAGE}件
            </Link>
          ) : (
            <span />
          )}
          <span className="kicker text-foreground/50">
            {currentPage} / {totalPages}
          </span>
          {currentPage < totalPages ? (
            <Link
              href={buildHref(selectedType, currentPage + 1)}
              className="text-brand-blue hover:underline"
            >
              古い{DISCLOSURES_PER_PAGE}件 ›
            </Link>
          ) : (
            <span />
          )}
        </nav>
      )}
    </>
  );
}
