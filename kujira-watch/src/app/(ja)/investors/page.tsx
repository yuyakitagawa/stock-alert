import type { Metadata } from "next";
import Link from "next/link";
import DealTypeBadge from "@/components/DealTypeBadge";
import { getAllFilers } from "@/lib/investors";
import { formatDate } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { DEAL_TYPES, type DealType } from "@/types/article";

export const revalidate = 3600;

const title = "投資家一覧";
const description =
  "EDINET大量保有報告書（5%ルール）を提出した機関投資家・アクティビストファンド・創業家の資産管理会社などの一覧。投資家別に保有銘柄・保有比率の推移を確認できます。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/investors` },
  openGraph: { title, description, url: `${SITE_URL}/investors` },
};

type Props = {
  searchParams: Promise<{ category?: string }>;
};

export default async function InvestorsPage({ searchParams }: Props) {
  const { category } = await searchParams;
  const filers = await getAllFilers();

  const counts = new Map<DealType, number>();
  for (const filer of filers) {
    counts.set(filer.category, (counts.get(filer.category) ?? 0) + 1);
  }
  const activeCategories = DEAL_TYPES.filter((c) => (counts.get(c) ?? 0) > 0);
  const selectedCategory =
    category && DEAL_TYPES.includes(category as DealType) ? (category as DealType) : null;
  const visibleFilers = selectedCategory
    ? filers.filter((f) => f.category === selectedCategory)
    : filers;

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
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">投資家一覧</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">投資家一覧</h1>
      <p className="mb-4 text-sm text-foreground/50">
        {SITE_NAME}がEDINET大量保有報告書から追跡している投資家（機関投資家・アクティビストファンド・
        創業家の資産管理会社など）{filers.length}件です。最終開示日が新しい順に並んでいます。
      </p>
      {filers.length > 0 && (
        <nav aria-label="カテゴリで絞り込む" className="kicker mb-6 flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <Link
            href="/investors"
            className={
              selectedCategory === null
                ? "font-bold text-brand-navy"
                : "text-brand-navy/60 transition-colors hover:text-brand-navy"
            }
          >
            すべて（{filers.length}件）
          </Link>
          {activeCategories.map((c) => (
            <Link
              key={c}
              href={`/investors?category=${encodeURIComponent(c)}`}
              className={
                selectedCategory === c
                  ? "font-bold text-brand-navy"
                  : "text-brand-navy/60 transition-colors hover:text-brand-navy"
              }
            >
              {c}（{counts.get(c)}件）
            </Link>
          ))}
        </nav>
      )}
      {visibleFilers.length === 0 ? (
        <p className="text-foreground/50">
          {filers.length === 0 ? "投資家データがまだありません。" : "該当する投資家がいません。"}
        </p>
      ) : (
        <ul className="divide-y divide-rule/50 border-t border-rule">
          {visibleFilers.map((filer) => (
            <li key={filer.filerName} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-3">
              <Link
                href={`/investors/${encodeURIComponent(filer.filerName)}`}
                className="font-medium text-brand-blue hover:underline"
              >
                {filer.filerName}
              </Link>
              <DealTypeBadge dealType={filer.category} />
              <span className="text-xs text-foreground/40">
                保有開示{filer.holdingCount}件・最終開示{formatDate(filer.latestDiscDate)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
