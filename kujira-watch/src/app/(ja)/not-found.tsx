import type { Metadata } from "next";
import Link from "next/link";
import { SITE_NAME } from "@/lib/site";

// 404はNext.jsの既定画面（ヘッダー・フッター・内部リンクなしの素のテキスト）だった。
// 記事の削除や開示の取り下げでURLが消えることが実際にあるため、行き止まりにせず
// 主要な索引ページへ戻せるようにする（2026-08-27のAdSense再監査での指摘）。
export const metadata: Metadata = {
  title: "ページが見つかりません",
  robots: { index: false, follow: true },
};

const LINKS = [
  { href: "/", label: "トップ", description: "最新の大量保有報告書と自社株買いの解説" },
  { href: "/articles", label: "記事一覧", description: "これまでの解説記事をすべて開示日順に" },
  { href: "/stocks", label: "銘柄一覧", description: "証券コードから銘柄ページを探す" },
  { href: "/investors", label: "投資家一覧", description: "提出者から保有の推移を追う" },
  { href: "/faq", label: "よくある質問", description: "大量保有報告書の読み方" },
];

export default function NotFound() {
  return (
    <div className="border-t border-rule bg-paper p-6 sm:p-10">
      <p className="mb-2 text-sm font-bold text-ink-tertiary">404</p>
      <h1 className="mb-4 text-2xl font-bold text-brand-navy sm:text-3xl">
        ページが見つかりません
      </h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        お探しのページは削除されたか、URLが変更された可能性があります。
        記載に誤りがあった記事は、開示原本と照合のうえ修正または削除しています。
        {SITE_NAME}の主なページは以下からご覧ください。
      </p>
      <ul className="space-y-3">
        {LINKS.map((link) => (
          <li key={link.href}>
            <Link href={link.href} className="font-bold text-brand-blue hover:underline">
              {link.label}
            </Link>
            <span className="ml-2 text-sm text-ink-tertiary">{link.description}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
