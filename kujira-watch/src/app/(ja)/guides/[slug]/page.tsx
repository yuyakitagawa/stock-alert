import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import AdUnit from "@/components/AdUnit";
import { GUIDES, findGuide, type GuideText } from "@/lib/guides";
import { formatDate, toDateAttr } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";

// 解説記事は lib/guides.ts に固定で持つので、ビルド時に全部わかる。
export function generateStaticParams() {
  return GUIDES.map((guide) => ({ slug: guide.slug }));
}
export const dynamicParams = false;

type Props = {
  params: Promise<{ slug: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const guide = findGuide(slug);
  if (!guide) return {};
  const url = `${SITE_URL}/guides/${slug}`;
  return {
    title: guide.title,
    description: guide.description,
    alternates: { canonical: url },
    openGraph: { title: guide.title, description: guide.description, url, type: "article" },
  };
}

const LINK_RE = /\[([^\]]+)\]\(([^)]+)\)/g;

// 段落内の [表示文字列](/path) だけをリンクにする（本文を lib/guides.ts に素の文字列で持つため）。
function Rich({ text }: { text: GuideText }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(LINK_RE)) {
    const index = match.index ?? 0;
    if (index > last) parts.push(text.slice(last, index));
    parts.push(
      <Link key={index} href={match[2]} className="text-brand-blue hover:underline">
        {match[1]}
      </Link>
    );
    last = index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}

export default async function GuidePage({ params }: Props) {
  const { slug } = await params;
  const guide = findGuide(slug);
  if (!guide) notFound();
  const url = `${SITE_URL}/guides/${slug}`;

  // 運営者は実名を出さない方針のため、著者は組織（サイト）として出す。
  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: guide.title,
    description: guide.description,
    datePublished: guide.published,
    dateModified: guide.updated ?? guide.published,
    author: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
    publisher: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
    mainEntityOfPage: url,
  };
  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "読み方ガイド", item: `${SITE_URL}/guides` },
      { "@type": "ListItem", position: 3, name: guide.title, item: url },
    ],
  };

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }} />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href="/guides" className="hover:text-brand-blue">読み方ガイド</Link>
        {" / "}
        <span className="text-ink-secondary">{guide.title}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold leading-snug text-brand-navy sm:text-3xl">{guide.title}</h1>
      <p className="mb-6 text-xs text-ink-tertiary">
        公開 <time dateTime={toDateAttr(guide.published)}>{formatDate(guide.published)}</time>
        {guide.updated && (
          <>
            {" ・ 更新 "}
            <time dateTime={toDateAttr(guide.updated)}>{formatDate(guide.updated)}</time>
          </>
        )}
      </p>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        <Rich text={guide.lead} />
      </p>
      {guide.sections.map((section) => (
        <section key={section.heading} className="mb-8">
          <h2 className="mb-3 text-xl font-bold text-brand-navy">{section.heading}</h2>
          {section.paragraphs.map((paragraph, i) => (
            <p key={i} className="mb-3 text-sm leading-relaxed text-ink-secondary">
              <Rich text={paragraph} />
            </p>
          ))}
        </section>
      ))}
      <p className="mb-8 border-t border-rule pt-4 text-xs leading-relaxed text-ink-tertiary">
        <Rich text={guide.sourceNote} />
      </p>
      {guide.related.length > 0 && (
        <section className="mb-8">
          <h2 className="mb-3 text-lg font-bold text-brand-navy">関連する記事・ページ</h2>
          <ul className="list-disc pl-5">
            {guide.related.map((item) => (
              <li key={item.href} className="mb-1 text-sm">
                <Link href={item.href} className="text-brand-blue hover:underline">{item.label}</Link>
              </li>
            ))}
          </ul>
        </section>
      )}
      <AdUnit placement="bottom" />
    </article>
  );
}
