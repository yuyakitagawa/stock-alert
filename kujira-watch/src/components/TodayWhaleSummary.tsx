import Link from "next/link";
import { formatDate, formatDealAmount } from "@/lib/format";

// TOPの冒頭に置く「今日のクジラ」サマリー。見出しは常に「今日の注目取引」で固定だが、
// 実際の最新開示が何日付なのか・その日に何件いくら動いたのかは記事カードを読むまで
// 分からなかった。開示日と件数・金額を最初に大きく出すことで、毎日更新されている
// ことが一目で分かるようにする（Discover・リピーター向けの鮮度表示）。
export default function TodayWhaleSummary({
  date,
  count,
  amount,
  buyCount,
  sellCount,
}: {
  date: string;
  count: number;
  amount: number;
  buyCount: number;
  sellCount: number;
}) {
  const dateOnly = date.slice(0, 10);
  return (
    <Link
      href={`/date/${dateOnly}`}
      className="group mb-8 block border-y-2 border-brand-navy py-4 transition-colors hover:bg-section-tint"
    >
      <p className="kicker text-brand-blue">今日のクジラ・{formatDate(date)}</p>
      <p className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-3xl font-bold leading-none text-brand-navy sm:text-4xl">{count}</span>
        <span className="text-sm text-foreground/60">件の開示</span>
        <span className="text-3xl font-bold leading-none text-brand-navy sm:text-4xl">
          {formatDealAmount(amount)}
        </span>
        <span className="text-sm text-foreground/60">
          （買い{buyCount}件・売り{sellCount}件）
        </span>
      </p>
      <p className="kicker mt-2 text-brand-blue transition-colors group-hover:text-brand-navy">
        この日の開示をすべて見る ›
      </p>
    </Link>
  );
}
