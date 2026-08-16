import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import AdUnit from "@/components/AdUnit";
import ActionButton from "@/components/ActionButton";
import FaqAccordionList from "@/components/FaqAccordionList";
import { FAQ_CATEGORIES, faqsByCategory, findCategory } from "@/lib/faqData";
import { SITE_NAME, SITE_URL } from "@/lib/site";

// カテゴリは固定9種でビルド時に全部わかるので事前生成する。
export function generateStaticParams() {
  return FAQ_CATEGORIES.map((category) => ({ category: category.id }));
}

type Props = {
  params: Promise<{ category: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { category: categoryId } = await params;
  const category = findCategory(categoryId);
  if (!category) return {};

  const faqs = faqsByCategory(categoryId);
  const title = `${category.label}｜よくある質問`;
  const description = `${category.label}に関するよくある質問と回答${faqs.length}件。${SITE_NAME}が大量保有報告書（5%ルール）の解説とあわせてまとめています。`;
  const url = `${SITE_URL}/faq/${categoryId}`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function FaqCategoryPage({ params }: Props) {
  const { category: categoryId } = await params;
  const category = findCategory(categoryId);
  if (!category) notFound();

  const faqs = faqsByCategory(categoryId);
  const url = `${SITE_URL}/faq/${categoryId}`;

  // FAQPage構造化データは「そのページに実際に表示されているQ&A」だけを載せる。
  // Googleのガイドラインで構造化データと可視コンテンツの一致が必須なため、
  // 他カテゴリの分をここに混ぜてはいけない。
  const faqJsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: faqs.map((faq) => ({
      "@type": "Question",
      name: faq.question,
      acceptedAnswer: { "@type": "Answer", text: faq.answer },
    })),
  };

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "よくある質問", item: `${SITE_URL}/faq` },
      { "@type": "ListItem", position: 3, name: category.label, item: url },
    ],
  };

  const others = FAQ_CATEGORIES.filter((c) => c.id !== categoryId);

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
        <Link href="/faq" className="hover:text-brand-blue">よくある質問</Link>
        {" / "}
        <span className="text-foreground/70">{category.label}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{category.label}</h1>
      <p className="mb-6 text-sm leading-relaxed text-foreground/70">
        {category.label}についてよく寄せられる質問と回答{faqs.length}件です。
        質問をタップすると回答が開きます。
      </p>

      <FaqAccordionList faqs={faqs} />

      <Box sx={{ mt: 6, pt: 4, borderTop: 1, borderColor: "divider" }}>
        <Typography variant="overline" sx={{ display: "block", mb: 1.5, color: "text.secondary" }}>
          ほかのカテゴリ
        </Typography>
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
          {others.map((other) => (
            <ActionButton key={other.id} href={`/faq/${other.id}`}>
              {other.label}（{faqsByCategory(other.id).length}）
            </ActionButton>
          ))}
        </Box>
      </Box>

      <AdUnit placement="bottom" />
    </article>
  );
}
