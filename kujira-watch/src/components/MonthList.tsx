import Link from "next/link";
import { formatDealAmount, formatMonth } from "@/lib/format";
import type { MonthSummary } from "@/lib/microcms";

// 月アーカイブのカード一覧。件数が十数件と少なく、1件ずつが独立した入口になるため
// グリッドで並べる（MUIのList/ListItemButtonは使わない。素のa.cardで十分かつ軽い）。
export default function MonthList({ months }: { months: MonthSummary[] }) {
  return (
    <ul className="card-grid">
      {months.map((m) => (
        <li key={m.month}>
          <Link href={`/monthly/${m.month}`} className="card">
            <span className="block font-bold text-brand-navy">{formatMonth(m.month)}</span>
            <span className="mt-1 block text-sm text-ink-tertiary">
              {m.count}件・{formatDealAmount(m.amount)}
            </span>
            <span className="kicker mt-2 block text-brand-blue">この月のまとめを見る ›</span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
