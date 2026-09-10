import type { Metadata } from "next";
import Link from "next/link";
import AdUnit from "@/components/AdUnit";
import ActionButton from "@/components/ActionButton";
import FilingDeadlineCalculator from "@/components/tools/FilingDeadlineCalculator";
import { chapterById, chapterNumber, chapterPath } from "@/lib/textbook";
import { toolBySlug, toolPath } from "@/lib/tools";
import { SITE_NAME, SITE_URL } from "@/lib/site";

const tool = toolBySlug("filing-deadline")!;
const url = `${SITE_URL}${toolPath(tool.slug)}`;

export const metadata: Metadata = {
  title: tool.title,
  description: tool.description,
  alternates: { canonical: url },
  openGraph: { title: tool.title, description: tool.description, url },
};

export default function FilingDeadlinePage() {
  const chapter = chapterById(tool.chapterId);
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
        { "@type": "ListItem", position: 2, name: "計算ツール", item: `${SITE_URL}/tools` },
        { "@type": "ListItem", position: 3, name: tool.title, item: url },
      ],
    },
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      name: tool.title,
      description: tool.description,
      url,
      applicationCategory: "FinanceApplication",
      operatingSystem: "All",
      inLanguage: "ja",
      isAccessibleForFree: true,
      offers: { "@type": "Offer", price: "0", priceCurrency: "JPY" },
      provider: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
    },
  ];

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href="/tools" className="hover:text-brand-blue">計算ツール</Link>
        {" / "}
        <span className="text-ink-secondary">{tool.title}</span>
      </nav>

      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{tool.title}</h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        株券等保有割合と取得日を入れると、大量保有報告書・変更報告書の提出義務の有無と提出期限が出ます。
        条文の「五日以内」は暦日ではないため、土日・国民の祝日・年末年始を除いて1日ずつ数えた過程も表示します。
      </p>

      <FilingDeadlineCalculator />

      <section className="mb-8">
        <h2 className="mb-3 text-xl font-bold text-brand-navy">計算の根拠</h2>
        <ul className="list-disc pl-5">
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">新規（大量保有報告書）</strong>: 金融商品取引法27条の23第1項。
            株券等保有割合が5%を超えた者は、超えることとなった日から5日以内に提出します。5%ちょうどは対象外です。
          </li>
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">変更報告書</strong>: 同法27条の25第1項。保有割合が1%以上増減した場合等に、
            同じく5日以内に提出します。5%以下になった場合は「5%以下になった旨」を報告して義務が終わります。
          </li>
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">数えない日</strong>: 条文の「五日」は日曜日その他政令で定める休日を算入しません。
            ここでは行政機関の休日に関する法律1条1項に合わせ、土曜・日曜・国民の祝日・12月29日〜1月3日を除いて数え、
            起算は初日不算入（民法140条）で取得日の翌日からとしています。
          </li>
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">特例報告</strong>: 同法27条の26。金融商品取引業者・銀行・保険会社等が重要提案行為等を
            目的とせず10%以下で保有する場合は、取引のたびではなく基準日ごとにまとめて報告できます。
          </li>
        </ul>
        <p className="mb-0 text-xs leading-relaxed text-ink-tertiary">
          祝日は国民の祝日に関する法律の規定（固定日・ハッピーマンデー・春分秋分・振替休日・国民の休日）から計算しています。
          春分の日・秋分の日は前年2月の官報で正式に決まるため、翌年以降の日付は見込みです。
          実際の提出にあたっては必ず条文と金融庁・EDINETの案内をご確認ください。本ツールは法律上の助言ではありません。
        </p>
      </section>

      {chapter && (
        <section className="mb-10 rounded-md border border-rule bg-section-tint p-4">
          <h2 className="mb-2 text-base font-bold text-brand-navy">この数字の意味を読む</h2>
          <p className="mb-3 text-sm leading-relaxed text-ink-secondary">
            提出義務と期限のルールは、教科書の第{chapterNumber(chapter.id)}章「{chapter.title}」で解説しています。
          </p>
          <div className="flex flex-wrap gap-3">
            <ActionButton href={chapterPath(chapter.id)}>{chapter.title}を読む</ActionButton>
            <ActionButton href="/articles">実際の開示を見る</ActionButton>
          </div>
        </section>
      )}

      <AdUnit placement="bottom" />
    </article>
  );
}
