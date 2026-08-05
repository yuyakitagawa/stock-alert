import type { Metadata } from "next";
import Link from "next/link";
import { SITE_NAME, SITE_URL } from "@/lib/site";

const title = "よくある質問";
const description = "大量保有報告書のしくみや本サイトの使い方についてのよくある質問。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/faq` },
};

// AI検索・LLM引用向けのFAQPage構造化データ。ページ本文（下記の一覧）と
// 一言一句一致させる（Googleのガイドラインで構造化データと可視コンテンツの一致が必須）。
const FAQS: { question: string; answer: string; render?: React.ReactNode }[] = [
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
  {
    question: "大量保有報告書と変更報告書の違いは何ですか？",
    answer:
      "新規に株式の5%以上を保有した場合は「大量保有報告書」、既に5%以上保有している投資家の保有比率が1%以上増減した場合は「変更報告書」として提出されます。本サイトではどちらも同様に取り上げています。",
  },
  {
    question: "誰が大量保有報告書を提出する義務がありますか？",
    answer:
      "上場株式の発行済株式総数の5%以上を保有することになった投資家（機関投資家・個人・ファンドなど）に提出義務があります。",
  },
  {
    question: "記事はどのくらいの頻度で更新されますか？",
    answer: "平日（取引日）に毎日、EDINETの新規開示をもとに記事を追加しています。",
  },
  {
    question: "直近の大口投資家の動きをまとめて見るにはどうすればいいですか？",
    answer:
      "トップページの「今週の動き」リンク、または「大口投資家の動き（直近7日間）」ページで、期間中の件数・合計推定金額とあわせて一覧できます。",
    render: (
      <>
        トップページの「今週の動き」リンク、または
        <Link href="/weekly" className="text-brand-blue hover:underline">
          「大口投資家の動き（直近7日間）」ページ
        </Link>
        で、期間中の件数・合計推定金額とあわせて一覧できます。
      </>
    ),
  },
  {
    question: "特定の銘柄の大量保有履歴を見るにはどうすればいいですか？",
    answer: "各記事の「銘柄」欄のリンクから、その銘柄の大量保有・自社株買い履歴をまとめたページに遷移できます。",
  },
  {
    question: "投資家の分類（バッジ）にはどんな種類がありますか？",
    answer:
      "個人・創業家の資産管理会社・アクティビスト・VC・国内外の資産運用会社など13種類に分類しています。詳しくは運営者情報ページの用語集で解説しています。",
    render: (
      <>
        個人・創業家の資産管理会社・アクティビスト・VC・国内外の資産運用会社など13種類に分類しています。詳しくは
        <Link href="/about#dealtype-glossary" className="text-brand-blue hover:underline">
          運営者情報ページの用語集
        </Link>
        で解説しています。
      </>
    ),
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

export default function FaqPage() {
  const url = `${SITE_URL}/faq`;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">{title}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      <p className="mb-6 text-sm leading-relaxed text-foreground/70">
        {SITE_NAME}の使い方や、大量保有報告書のしくみについてよく寄せられる質問をまとめました。
      </p>
      <dl className="space-y-5 text-sm">
        {FAQS.map((faq) => (
          <div key={faq.question}>
            <dt className="font-semibold text-brand-navy">{faq.question}</dt>
            <dd className="mt-1 text-foreground/70">{faq.render ?? faq.answer}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
}
