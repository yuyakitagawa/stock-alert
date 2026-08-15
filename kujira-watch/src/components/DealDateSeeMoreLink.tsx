import ActionButton from "./ActionButton";
import type { Locale } from "@/lib/i18n";

// /en/date/[date] は現状未実装のため、英語版ではリンクを出さない。
// その日の記事カードを見せた後の導線として、次の日付見出しの直前（右寄せ）に置く。
export default function DealDateSeeMoreLink({ date, locale = "ja" }: { date: string; locale?: Locale }) {
  if (locale === "en") return null;

  return (
    <div className="mt-4 text-right">
      <ActionButton href={`/date/${date}`}>この日の記事を見る</ActionButton>
    </div>
  );
}
