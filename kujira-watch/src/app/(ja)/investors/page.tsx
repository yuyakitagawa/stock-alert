import type { Metadata } from "next";
import Link from "next/link";
import DealTypeBadge from "@/components/DealTypeBadge";
import { getAllFilers } from "@/lib/investors";
import { formatDate } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";

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

export default async function InvestorsPage() {
  const filers = await getAllFilers();

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
      <p className="mb-6 text-sm text-foreground/50">
        {SITE_NAME}がEDINET大量保有報告書から追跡している投資家（機関投資家・アクティビストファンド・
        創業家の資産管理会社など）{filers.length}件です。最終開示日が新しい順に並んでいます。
      </p>
      {filers.length === 0 ? (
        <p className="text-foreground/50">投資家データがまだありません。</p>
      ) : (
        <ul className="divide-y divide-rule/50 border-t border-rule">
          {filers.map((filer) => (
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
