import Link from "next/link";
import type { Locale } from "@/lib/i18n";

// /en/date/[date] は現状未実装のため、英語版では日付見出しのみ表示しリンクは出さない。
export default function DealDateHeading({
  date,
  label,
  locale = "ja",
}: {
  date: string;
  label: string;
  locale?: Locale;
}) {
  return (
    <div className="mb-5">
      <h2 className="flex items-center gap-3 text-base font-bold text-brand-navy">
        {label}
        <span aria-hidden className="h-px flex-1 bg-rule" />
      </h2>
      {locale === "ja" && (
        <Link
          href={`/date/${date}`}
          className="mt-1 inline-block text-xs text-brand-blue transition-colors hover:text-brand-navy hover:underline"
        >
          この日の記事を見る ›
        </Link>
      )}
    </div>
  );
}
