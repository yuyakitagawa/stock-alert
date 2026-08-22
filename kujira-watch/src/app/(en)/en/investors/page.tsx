import type { Metadata } from "next";
import Link from "next/link";
import { getAllFilers, investorPath } from "@/lib/investors";
import { displayFilerName, formatDate } from "@/lib/format";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import { SITE_NAME_EN, SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

export const revalidate = 3600;

// 英語版はこれまで記事とカテゴリしか無く、「誰がクジラなのか」を一覧で見る手段が無かった
// （2026-08-19のユーザーインタビュー・英語圏セグメントの継続要望）。
// 英訳記事は価値の高い開示だけに絞る方針（isIndexableEnArticle）のため、投資家の一覧は
// 記事ではなくEDINET開示の集計（edinet_filer_summary）から作る。
// 全2,900件を並べると重いので、直近の提出があるアクティブな投資家の上位だけを載せる。
const TOP_COUNT = 100;

const url = `${SITE_URL}/en/investors`;
const title = "Top Whales — Most Active Large-Shareholding Filers";
const description =
  "The investors filing the most large-shareholding reports (5% rule) on the Tokyo market: activist funds, foreign asset managers, founding families and corporate holders, with their filing counts and latest filing dates.";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: url, languages: { ja: `${SITE_URL}/investors`, en: url } },
  openGraph: { title, description, url },
};

export default async function EnInvestorsPage() {
  const filers = await getAllFilers().catch(() => []);
  const top = [...filers]
    .sort((a, b) => b.holdingCount - a.holdingCount || b.latestDiscDate.localeCompare(a.latestDiscDate))
    .slice(0, TOP_COUNT);

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: title,
    itemListElement: top.slice(0, 30).map((filer, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: displayFilerName(filer.filerName),
      url: `${SITE_URL}${investorPath(filer.filerId, filer.filerName)}`,
    })),
  };

  return (
    <div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
      />
      <nav aria-label="Breadcrumb" className="mb-4 text-xs text-foreground/50">
        <Link href="/en" className="hover:text-brand-blue">
          Top
        </Link>
        {" / "}
        <span className="text-foreground/70">Top Whales</span>
      </nav>
      <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">Top Whales</h1>
      <p className="mt-2 text-sm leading-relaxed text-foreground/70">
        Investors ranked by how many EDINET large-shareholding reports (the Japanese 5% rule) they
        have filed. Every filing is a public disclosure of a stake of 5% or more — these are the
        players whose moves the Japanese market watches.
      </p>
      {/* 英語版は価値の高い記事だけに絞っているため、個別の投資家ページは日本語のみ。
          英語話者が行き止まりに感じないよう、リンク先が日本語であることを先に伝える。 */}
      <p className="mt-2 text-xs text-foreground/50">
        Investor detail pages are published in Japanese only. {SITE_NAME_EN} translates the
        highest-impact filings into English; the full archive lives on the{" "}
        <Link href="/" className="text-brand-blue hover:underline" hrefLang="ja">
          Japanese site
        </Link>
        .
      </p>

      {top.length === 0 ? (
        <p className="mt-8 text-sm text-foreground/60">No filer data available right now.</p>
      ) : (
        <ol className="mt-6 space-y-2">
          {top.map((filer, index) => (
            <li key={filer.filerName} className="card text-sm">
              <span className="mr-2 text-xs font-bold text-foreground/40">{index + 1}</span>
              <Link
                href={investorPath(filer.filerId, filer.filerName)}
                hrefLang="ja"
                className="font-medium text-brand-blue hover:underline"
              >
                {displayFilerName(filer.filerName)}
              </Link>
              <span className="ml-2 text-xs text-foreground/50">
                {DEAL_TYPE_EN[filer.category]?.label ?? filer.category}
              </span>
              <span className="mt-1 block text-xs text-foreground/60">
                {filer.holdingCount} filings · latest {formatDate(filer.latestDiscDate, "en")}
              </span>
            </li>
          ))}
        </ol>
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}
