import type { Metadata } from "next";
import Link from "next/link";
import AdUnit from "@/components/AdUnit";
import ActionButton from "@/components/ActionButton";
import { TEXTBOOK_CHAPTERS, chapterPath } from "@/lib/textbook";
import { TOOLS, toolPath } from "@/lib/tools";
import { SITE_NAME, SITE_URL } from "@/lib/site";

const url = `${SITE_URL}/textbook`;
const title = "大量保有報告書の教科書";
const description =
  "上場企業の大量保有報告書（5%ルール）を自力で読めるようになるための全6章。制度の目的、提出義務と期限、報告書の読み方、提出者の分類、自社株買い、開示を使った検証の手順までを順番に解説し、各章で当日の実際の開示を確認できます。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: url },
  openGraph: { title, description, url },
};

export default function TextbookPage() {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
        { "@type": "ListItem", position: 2, name: title, item: url },
      ],
    },
    // 章の順路そのものを構造化データで示す。AI検索が「この教科書は何章構成か」を
    // 本文の解析ではなく宣言から取れるようにするため。
    {
      "@context": "https://schema.org",
      "@type": "Course",
      name: title,
      description,
      url,
      provider: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
      hasPart: TEXTBOOK_CHAPTERS.map((chapter, index) => ({
        "@type": "LearningResource",
        position: index + 1,
        name: chapter.title,
        description: chapter.description,
        url: `${SITE_URL}${chapterPath(chapter.id)}`,
      })),
    },
  ];

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-ink-secondary">{title}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        EDINETに毎日出てくる大量保有報告書を、解説なしで読めるようになるための全
        {TEXTBOOK_CHAPTERS.length}章です。制度の説明で終わらせず、各章の最後にその日実際に出ている開示を並べています。
        読んだ内容を自分の持ち株で確かめるための計算ツールも用意しました。
      </p>

      <ol className="mb-10 list-none p-0">
        {TEXTBOOK_CHAPTERS.map((chapter, index) => (
          <li key={chapter.id} className="mb-5 border-b border-rule pb-5">
            <p className="mb-1 text-2xs font-bold text-ink-muted">第{index + 1}章</p>
            <h2 className="mb-2 text-xl font-bold text-brand-navy">
              <Link href={chapterPath(chapter.id)} className="hover:text-brand-blue hover:underline">
                {chapter.title}
              </Link>
            </h2>
            <p className="mb-3 text-sm leading-relaxed text-ink-secondary">{chapter.lead}</p>
            <ul className="mb-3 list-disc pl-5">
              {chapter.goals.map((goal) => (
                <li key={goal} className="mb-1 text-xs text-ink-tertiary">{goal}</li>
              ))}
            </ul>
            <ActionButton href={chapterPath(chapter.id)}>第{index + 1}章を読む</ActionButton>
          </li>
        ))}
      </ol>

      <section className="mb-10">
        <h2 className="mb-3 text-xl font-bold text-brand-navy">計算ツール</h2>
        <p className="mb-3 text-sm leading-relaxed text-ink-secondary">
          教科書で出てくる数字を、自分が見ている銘柄の数字で確かめられます。入力値は送信せず、
          すべてブラウザ内で計算します。
        </p>
        <ul className="list-disc pl-5">
          {TOOLS.map((tool) => (
            <li key={tool.slug} className="mb-2 text-sm text-ink-secondary">
              <Link href={toolPath(tool.slug)} className="text-brand-blue hover:underline">
                {tool.title}
              </Link>
              <span className="ml-1 text-xs text-ink-tertiary">{tool.summary}</span>
            </li>
          ))}
        </ul>
      </section>

      <p className="mb-8 text-xs leading-relaxed text-ink-tertiary">
        個別の疑問から探したいときは<Link href="/faq" className="text-brand-blue hover:underline">よくある質問</Link>
        に用語・制度・使い方のQ&amp;Aをカテゴリ別にまとめています。
      </p>

      <AdUnit placement="bottom" />
    </article>
  );
}
