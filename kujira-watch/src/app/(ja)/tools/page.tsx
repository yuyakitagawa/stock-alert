import type { Metadata } from "next";
import Link from "next/link";
import AdUnit from "@/components/AdUnit";
import ActionButton from "@/components/ActionButton";
import { TOOLS, toolPath } from "@/lib/tools";
import { chapterById, chapterPath } from "@/lib/textbook";
import { SITE_NAME, SITE_URL } from "@/lib/site";

const url = `${SITE_URL}/tools`;
const title = "計算ツール";
const description =
  "大量保有報告書の提出期限・提出義務の判定、自社株買いの取得枠が発行済株式やEPSに与える影響など、開示を読むときに必要な計算をその場で行える無料ツールです。入力値は送信せず、すべてブラウザ内で計算します。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: url },
  openGraph: { title, description, url },
};

export default function ToolsPage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-ink-secondary">{title}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        {SITE_NAME}が毎日集めている開示は「起きたこと」ですが、自分が見ている銘柄で同じ計算をやり直す手段が必要になります。
        ここでは開示を読むときに使う計算をその場で行えるようにしました。入力した数字はサーバーに送らず、
        すべてブラウザの中だけで計算します。
      </p>

      <ul className="mb-10 list-none p-0">
        {TOOLS.map((tool) => {
          const chapter = chapterById(tool.chapterId);
          return (
            <li key={tool.slug} className="mb-5 border-b border-rule pb-5">
              <h2 className="mb-2 text-xl font-bold text-brand-navy">
                <Link href={toolPath(tool.slug)} className="hover:text-brand-blue hover:underline">
                  {tool.title}
                </Link>
              </h2>
              <p className="mb-3 text-sm leading-relaxed text-ink-secondary">{tool.summary}</p>
              <p className="mb-3 text-xs text-ink-tertiary">入力: {tool.inputs.join(" / ")}</p>
              <div className="mb-3 flex flex-wrap gap-3">
                <ActionButton href={toolPath(tool.slug)}>ツールを開く</ActionButton>
              </div>
              {chapter && (
                <p className="mb-0 text-xs text-ink-tertiary">
                  数字の意味は
                  <Link href={chapterPath(chapter.id)} className="text-brand-blue hover:underline">
                    教科書「{chapter.title}」
                  </Link>
                  で解説しています。
                </p>
              )}
            </li>
          );
        })}
      </ul>

      <AdUnit placement="bottom" />
    </article>
  );
}
