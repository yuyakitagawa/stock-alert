import type { Metadata } from "next";
import Link from "next/link";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { DEAL_TYPES } from "@/types/article";

const title = "運営者情報・免責事項";
const description = "本サイトの運営方針、データソース、免責事項について。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/about` },
};

// AI検索・LLM引用向けのFAQPage構造化データ。ページ本文（下記「よくある質問」節）と
// 一言一句一致させる（Googleのガイドラインで構造化データと可視コンテンツの一致が必須）。
const FAQS: { question: string; answer: string }[] = [
  {
    question: "大量保有報告書とは何ですか？",
    answer:
      "上場企業の株式を5%以上保有した投資家が、金融庁の開示システムEDINETに提出する法定書類です（5%ルール）。保有比率が1%以上増減した場合は、変更報告書として再提出されます。",
  },
  {
    question: "「クジラ」とは何ですか？",
    answer:
      "相場を動かすほどの資金力を持つ機関投資家・アクティビストファンド・創業家の資産管理会社など、大口投資家を指す金融業界の俗称です。",
  },
  {
    question: "記事の金額規模（推定取得金額）はどうやって算出していますか？",
    answer:
      "EDINET開示自体には金額の記載がないため、発行済株式数×株価×保有比率の変化から独自に推定しています。正確な取得価格を保証するものではありません。",
  },
  {
    question: "このサイトの情報は投資助言ですか？",
    answer:
      "いいえ。本サイトの内容は情報提供を目的としたものであり、特定の銘柄や投資判断を推奨・勧誘するものではありません。",
  },
  {
    question: "記事はどうやって作成されていますか？",
    answer: "AIがEDINET開示等の事実情報から生成した上で、必要に応じて人が内容を確認・修正しています。",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: FAQS.map((faq) => ({
    "@type": "Question",
    name: faq.question,
    acceptedAnswer: { "@type": "Answer", text: faq.answer },
  })),
};

export default function AboutPage() {
  return (
    <article className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-200">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
      />
      <h1 className="mb-6 text-2xl font-bold text-brand-navy">{title}</h1>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">大口投資家の動きとは</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          「大口投資家の動き」とは、機関投資家・アクティビストファンド・創業家の資産管理会社・
          自社株買いなど、相場を動かすほどの資金力を持つ投資主体（クジラ）が、上場企業の株式を
          いつ・どれくらいの規模で売買したかを指します。日本では、株式の5%以上を保有した投資家は
          EDINET（金融庁の開示システム）に大量保有報告書の提出が義務付けられており（5%ルール）、
          保有比率が1%以上増減した場合は変更報告書として再提出されます。{SITE_NAME}はこの公開情報を
          毎日追跡し、銘柄別・投資家分類別に整理して解説しています。直近の動きは
          <Link href="/weekly" className="text-brand-blue hover:underline">
            「今週の動き」ページ
          </Link>
          でまとめて確認できます。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">このサイトについて</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          {SITE_NAME}は、EDINET大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
          インサイダー・自社株買いといった「クジラ」（相場を動かすほどの資金力を持つ大口投資家を指す
          金融業界の俗称）の動きを監視し、解説するブログです。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">データソースと更新方法</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-gray-700">
          <li>取引情報はEDINET大量保有報告書（買い方向のみ）を日次で取得しています。</li>
          <li>
            金額規模は、発行済株式数×株価×保有比率変化から算出した推定値です。EDINET開示自体には
            金額の記載がないため、正確な金額を保証するものではありません。
          </li>
          <li>記事はAIが事実情報から生成した上で、必要に応じて人が内容を確認・修正しています。</li>
          <li>出典元（適時開示・プレスリリース等）が判明している記事には、記事内にリンクを掲載しています。</li>
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">よくある質問</h2>
        <dl className="space-y-4 text-sm">
          {FAQS.map((faq) => (
            <div key={faq.question}>
              <dt className="font-semibold text-brand-navy">{faq.question}</dt>
              <dd className="mt-1 text-gray-700">{faq.answer}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section id="dealtype-glossary" className="mb-6 scroll-mt-20">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">投資家分類（用語集）</h2>
        <p className="mb-3 text-sm leading-relaxed text-gray-700">
          記事に付いているバッジは、大量保有報告書の提出者を以下のいずれかに分類したものです。
        </p>
        <dl className="space-y-3 text-sm">
          {DEAL_TYPES.map((dealType) => (
            <div key={dealType}>
              <dt className="font-semibold text-brand-navy">{dealType}</dt>
              <dd className="text-gray-700">{DEAL_TYPE_DESCRIPTIONS[dealType]}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section>
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">免責事項</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          本サイトの内容は情報提供を目的としたものであり、特定の銘柄や投資判断を推奨・勧誘するものではありません。
          掲載情報の正確性・完全性を保証するものではなく、本サイトの情報に基づいて被ったいかなる損害についても
          運営者は責任を負いません。投資に関する最終判断はご自身の責任で行ってください。
        </p>
      </section>
    </article>
  );
}
