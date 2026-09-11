import type { Metadata } from "next";
import Link from "next/link";
import AdUnit from "@/components/AdUnit";
import { GUIDES_NEWEST_FIRST } from "@/lib/guides";
import { formatDate, toDateAttr } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";

const title = "大量保有報告書の読み方ガイド";
const description =
  "大量保有報告書の「保有目的」「短期大量譲渡」「特例報告」「潜在株式を含む保有比率」など、開示を読むときに誤解しやすい点を、実際の開示例とデータで解説します。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/guides` },
  openGraph: { title, description, url: `${SITE_URL}/guides` },
};

// 開示1件ごとの自動生成記事とは別に、編集部が開示を横断して書いた解説記事の一覧。
// 2026-09のAdSense再審査対応（「有用性の低いコンテンツ」）で、サイト全体が開示ごとの
// 定型記事に偏っていたため、実際の開示例とEDINET全件の集計に基づく読み物を置く。
export default function GuidesPage() {
  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: `${SITE_URL}/guides` },
    ],
  };

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-ink-secondary">{title}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        {SITE_NAME}が毎日追っている大量保有報告書は、数字だけを見ると誤解しやすい開示です。
        保有比率が下がっても売ったとは限らず、証券会社の5%超えは投資判断とは限りません。
        ここでは、実際に提出された開示を例に、読み違えやすい点を一つずつ解説します。
      </p>
      <ul className="m-0 list-none p-0">
        {GUIDES_NEWEST_FIRST.map((guide) => (
          <li key={guide.slug} className="mb-6 border-b border-rule pb-6">
            <h2 className="mb-1 text-xl font-bold text-brand-navy">
              <Link href={`/guides/${guide.slug}`} className="hover:text-brand-blue hover:underline">
                {guide.title}
              </Link>
            </h2>
            <p className="mb-2 text-xs text-ink-tertiary">
              <time dateTime={toDateAttr(guide.published)}>{formatDate(guide.published)}</time>
            </p>
            <p className="m-0 text-sm leading-relaxed text-ink-secondary">{guide.description}</p>
          </li>
        ))}
      </ul>
      <AdUnit placement="bottom" />
    </article>
  );
}
