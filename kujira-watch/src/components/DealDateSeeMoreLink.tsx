import Link from "next/link";
import type { Locale } from "@/lib/i18n";

// /en/date/[date] は現状未実装のため、英語版ではリンクを出さない。
// その日の記事カードを見せた後の導線として、次の日付見出しの直前（右寄せ）に置く。
export default function DealDateSeeMoreLink({ date, locale = "ja" }: { date: string; locale?: Locale }) {
  if (locale === "en") return null;

  return (
    <div className="mt-4 text-right">
      <Link
        href={`/date/${date}`}
        className="inline-block text-xs text-brand-blue transition-colors hover:text-brand-navy hover:underline"
      >
        この日の記事を見る ›
      </Link>
    </div>
  );
}
