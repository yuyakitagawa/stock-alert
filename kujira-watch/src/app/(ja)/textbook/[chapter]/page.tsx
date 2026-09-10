import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import AdUnit from "@/components/AdUnit";
import ActionButton from "@/components/ActionButton";
import RelatedArticles from "@/components/RelatedArticles";
import { getArticleList } from "@/lib/microcms";
import { CATEGORY_COLORS, faqsByCategory, findCategory } from "@/lib/faqData";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import {
  TEXTBOOK_CHAPTERS,
  chapterById,
  chapterNeighbors,
  chapterNumber,
  chapterPath,
} from "@/lib/textbook";
import { toolBySlug, toolPath } from "@/lib/tools";
import type { ArticleContent, DealType } from "@/types/article";

// 本文は固定だが「この章の実例」は最新の開示から引くため、1時間ごとに作り直す。
// 制度の説明ページを毎日更新されるデータページと同じ鮮度に保つのがこの教科書の狙い。
export const revalidate = 3600;

// 章は6つで固定。
export function generateStaticParams() {
  return TEXTBOOK_CHAPTERS.map((chapter) => ({ chapter: chapter.id }));
}

// 章末に出す実例記事の件数。
const EXAMPLE_LIMIT = 3;

type Props = { params: Promise<{ chapter: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { chapter: chapterId } = await params;
  const chapter = chapterById(chapterId);
  if (!chapter) return {};

  const title = `第${chapterNumber(chapter.id)}章 ${chapter.title}｜大量保有報告書の教科書`;
  const url = `${SITE_URL}${chapterPath(chapter.id)}`;
  return {
    title,
    description: chapter.description,
    alternates: { canonical: url },
    openGraph: { title, description: chapter.description, url },
  };
}

// microCMSが一時的に落ちても、制度の解説である本文までは必ず出す。
async function loadExamples(dealType?: DealType): Promise<ArticleContent[]> {
  try {
    const { contents } = await getArticleList({ limit: EXAMPLE_LIMIT, dealType });
    return contents;
  } catch {
    return [];
  }
}

export default async function TextbookChapterPage({ params }: Props) {
  const { chapter: chapterId } = await params;
  const chapter = chapterById(chapterId);
  if (!chapter) notFound();

  const number = chapterNumber(chapter.id);
  const url = `${SITE_URL}${chapterPath(chapter.id)}`;
  const { prev, next } = chapterNeighbors(chapter.id);
  const examples = await loadExamples(chapter.example.dealType);
  const tools = chapter.toolSlugs.map(toolBySlug).filter((tool) => tool !== undefined);

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
        { "@type": "ListItem", position: 2, name: "大量保有報告書の教科書", item: `${SITE_URL}/textbook` },
        { "@type": "ListItem", position: 3, name: chapter.title, item: url },
      ],
    },
    {
      "@context": "https://schema.org",
      "@type": "LearningResource",
      name: chapter.title,
      description: chapter.description,
      url,
      learningResourceType: "解説",
      inLanguage: "ja",
      teaches: chapter.goals,
      isPartOf: { "@type": "Course", name: "大量保有報告書の教科書", url: `${SITE_URL}/textbook` },
      provider: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
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
        <Link href="/textbook" className="hover:text-brand-blue">大量保有報告書の教科書</Link>
        {" / "}
        <span className="text-ink-secondary">{chapter.title}</span>
      </nav>

      <p className="mb-1 text-2xs font-bold text-ink-muted">第{number}章</p>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{chapter.title}</h1>
      <p className="mb-6 text-sm leading-relaxed text-ink-secondary">{chapter.lead}</p>

      <section className="mb-8 rounded-md border border-rule bg-section-tint p-4">
        <h2 className="mb-2 text-xs font-bold text-ink-secondary">この章でわかること</h2>
        <ul className="m-0 list-disc pl-5">
          {chapter.goals.map((goal) => (
            <li key={goal} className="mb-1 text-sm text-ink-secondary">{goal}</li>
          ))}
        </ul>
      </section>

      {chapter.sections.map((section) => (
        <section key={section.heading} className="mb-8">
          <h2 className="mb-3 text-xl font-bold text-brand-navy">{section.heading}</h2>
          {section.paragraphs.map((paragraph) => (
            <p key={paragraph.slice(0, 24)} className="mb-4 text-base leading-relaxed text-ink">
              {paragraph}
            </p>
          ))}
        </section>
      ))}

      {tools.length > 0 && (
        <section className="mb-10 rounded-md border border-rule bg-section-tint p-4">
          <h2 className="mb-2 text-base font-bold text-brand-navy">自分の数字で確かめる</h2>
          <ul className="m-0 list-disc pl-5">
            {tools.map((tool) => (
              <li key={tool.slug} className="mb-1 text-sm text-ink-secondary">
                <Link href={toolPath(tool.slug)} className="text-brand-blue hover:underline">
                  {tool.title}
                </Link>
                <span className="ml-1 text-xs text-ink-tertiary">{tool.summary}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* この教科書が固定ページと違うところ。制度の説明を読んだ直後に、いま出ている開示を見る。 */}
      <RelatedArticles
        title={`${chapter.example.label}で確かめる`}
        lead={chapter.example.note}
        articles={examples}
      />

      <section className="mb-10">
        <h2 className="mb-3 text-xl font-bold text-brand-navy">この章に関係するデータページ</h2>
        <ul className="list-disc pl-5">
          {chapter.relatedPages.map((page) => (
            <li key={page.href} className="mb-1 text-sm">
              <Link href={page.href} className="text-brand-blue hover:underline">{page.label}</Link>
            </li>
          ))}
        </ul>
      </section>

      <section className="mb-10">
        <h2 className="mb-3 text-xl font-bold text-brand-navy">もっと細かい疑問から探す</h2>
        <ul className="list-disc pl-5">
          {chapter.faqCategories.map((categoryId) => {
            const category = findCategory(categoryId);
            if (!category) return null;
            const count = faqsByCategory(categoryId).length;
            return (
              <li key={categoryId} className="mb-1 text-sm">
                <Link
                  href={`/faq/${categoryId}`}
                  className="hover:underline"
                  style={{ color: CATEGORY_COLORS[categoryId] ?? "var(--ink-secondary)" }}
                >
                  {category.label}
                </Link>
                <span className="ml-1 text-xs text-ink-tertiary">Q&amp;A {count}件</span>
              </li>
            );
          })}
        </ul>
      </section>

      <nav aria-label="章の移動" className="mb-8 flex flex-wrap gap-3 border-t border-rule pt-6">
        {prev && <ActionButton href={chapterPath(prev.id)}>← 第{chapterNumber(prev.id)}章 {prev.title}</ActionButton>}
        {next && <ActionButton href={chapterPath(next.id)}>第{chapterNumber(next.id)}章 {next.title} →</ActionButton>}
        {!next && <ActionButton href="/textbook">目次に戻る</ActionButton>}
      </nav>

      <AdUnit placement="bottom" />
    </article>
  );
}
