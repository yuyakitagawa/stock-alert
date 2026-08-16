import type { Metadata } from "next";
import Link from "next/link";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import AdUnit from "@/components/AdUnit";
import ActionButton from "@/components/ActionButton";
import { CATEGORY_COLORS, FAQ_CATEGORIES, FAQS, faqsByCategory } from "@/lib/faqData";
import { SITE_NAME, SITE_URL } from "@/lib/site";

// ハブに出す質問のサンプル件数。全文はカテゴリページ側に置く。
const PREVIEW_COUNT = 5;

const title = "よくある質問";
const description =
  "大量保有報告書・5%ルールのしくみから、投資初心者・中級者が疑問に思うポイント、本サイトの使い方まで、よくある質問と回答をカテゴリ別にまとめました。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/faq` },
  openGraph: { title, description, url: `${SITE_URL}/faq` },
};

// このページはカテゴリへの入口（ハブ）。
//
// 以前は全502件のQ&Aを1ページに並べていて、HTMLが1.57MB(gzip 241KB)まで膨らんでいた。
// 内訳は可視HTML 873KB・RSCペイロード 471KB・FAQPage構造化データ 225KB で、
// 同じ本文が1つの文書に3回入っていた（クライアントコンポーネントへのprops渡しと
// 構造化データで二重・三重になる）。カテゴリ別ページに分割して解消している。
//
// ここに回答本文を置かないのは意図的。表示していないQ&AをFAQPage構造化データに
// 載せるのはGoogleのガイドライン違反（構造化データと可視コンテンツの一致が必須）に
// なるため、構造化データも各カテゴリページ側に置いている。
export default function FaqPage() {
  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: `${SITE_URL}/faq` },
    ],
  };

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
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
      <p className="mb-8 text-sm leading-relaxed text-foreground/70">
        {SITE_NAME}の使い方、大量保有報告書（5%ルール）のしくみ、投資家分類の意味から、
        投資初心者・中級者の方からよく寄せられる疑問まで、全{FAQS.length}件をカテゴリ別に
        まとめました。気になるカテゴリを選んでご覧ください。
      </p>

      {FAQ_CATEGORIES.map((category) => {
        const faqs = faqsByCategory(category.id);
        if (faqs.length === 0) return null;
        const color = CATEGORY_COLORS[category.id] ?? "text.secondary";

        return (
          <Box key={category.id} sx={{ mb: 5, pb: 4, borderBottom: 1, borderColor: "divider" }}>
            <Typography
              variant="overline"
              sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.5, color }}
            >
              <Box
                component="span"
                aria-hidden
                sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "currentColor" }}
              />
              {faqs.length}件
            </Typography>
            <h2 className="mb-3 text-lg font-bold text-brand-navy">
              <Link href={`/faq/${category.id}`} className="hover:text-brand-blue hover:underline">
                {category.label}
              </Link>
            </h2>
            <ul className="mb-4 list-disc pl-5">
              {faqs.slice(0, PREVIEW_COUNT).map((faq) => (
                <li key={faq.question} className="mb-1 text-sm text-foreground/70">
                  <Link href={`/faq/${category.id}`} className="hover:text-brand-blue hover:underline">
                    {faq.question}
                  </Link>
                </li>
              ))}
            </ul>
            <ActionButton href={`/faq/${category.id}`}>
              {category.label}のQ&amp;Aをすべて見る
            </ActionButton>
          </Box>
        );
      })}

      <AdUnit placement="bottom" />
    </article>
  );
}
