import type { Metadata } from "next";
import Link from "next/link";
import { SITE_URL } from "@/lib/site";

export const metadata: Metadata = {
  title: "Page not found",
  robots: { index: false, follow: true },
};

// 英訳が無い記事や削除済みのURLで出る。行き止まりにせず、英語版トップと
// 日本語版（全記事がある側）へ戻せるようにする。
export default function EnNotFound() {
  return (
    <div className="border-t border-rule bg-paper p-6 sm:p-10">
      <p className="mb-2 text-sm font-bold text-ink-tertiary">404</p>
      <h1 className="mb-4 text-2xl font-bold text-brand-navy sm:text-3xl">Page not found</h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        This page may have been removed, or it has no English edition. Only a subset of articles is
        available in English; the Japanese edition carries every disclosure.
      </p>
      <ul className="space-y-3">
        <li>
          <Link href="/" className="font-bold text-brand-blue hover:underline">Latest articles</Link>
          <span className="ml-2 text-sm text-ink-tertiary">English edition top page</span>
        </li>
        <li>
          <a href={SITE_URL} hrefLang="ja" className="font-bold text-brand-blue hover:underline">Japanese edition</a>
          <span className="ml-2 text-sm text-ink-tertiary">All disclosures, stock pages, and investor pages</span>
        </li>
      </ul>
    </div>
  );
}
