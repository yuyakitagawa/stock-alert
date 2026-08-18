import type { Metadata } from "next";
import Link from "next/link";
import MonthList from "@/components/MonthList";
import { formatMonth } from "@/lib/format";
import { getAllMonthsForIndex } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

export const revalidate = 60;

const url = `${SITE_URL}/monthly`;
const title = "月別アーカイブ";

export const metadata: Metadata = {
  title,
  description:
    "EDINET大量保有報告書をもとにした大口投資家の動きを月ごとにまとめたアーカイブ。各月の開示件数・推定取引金額から、その月に動いた投資家・銘柄をたどれます。",
  alternates: { canonical: url },
  openGraph: { title, url },
};

// 月別アーカイブの入口。取引日別ページ(/date/[date])は日数分だけ増える一方で、
// 直近7日ぶんが/weeklyから張られるだけで古い日付は内部リンクが切れていた
// （サイトマップにしか載らない孤立ページになっていた）。月ハブを挟むことで
// 全ての取引日別ページが「ヘッダー → 月別アーカイブ → 各月 → 各日」で辿れるようにする。
export default async function MonthlyIndexPage() {
  const months = await getAllMonthsForIndex();

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: title,
    itemListElement: months.map((m, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: `${formatMonth(m.month)}の大口投資家の動き`,
      url: `${SITE_URL}/monthly/${m.month}`,
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
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">月別アーカイブ</h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}が公開した大口投資家の動きを月ごとにまとめています。
          各月のページでは、その月に動いた投資家・銘柄のランキングと日別の一覧を確認できます。
        </p>
      </div>
      {months.length === 0 ? (
        <p className="text-sm text-foreground/60">まだ記事がありません。</p>
      ) : (
        <MonthList months={months} />
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}
