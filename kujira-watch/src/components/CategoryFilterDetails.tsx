import Link from "next/link";
import { CATEGORIES } from "@/types/article";

// TOPページの見出し直下に置くカテゴリ絞り込み。以前はHeader(全ページ共通)に
// 常設していたが、日付一覧の直前に移した方が「見出し→フィルター→一覧」の
// 導線として分かりやすいため専用コンポーネントに切り出した。
export default function CategoryFilterDetails() {
  return (
    <details className="group mb-8 border-b border-rule pb-4">
      <summary className="kicker cursor-pointer list-none text-brand-navy/50 transition-colors hover:text-brand-navy [&::-webkit-details-marker]:hidden">
        カテゴリで絞り込む
        <span aria-hidden className="ml-1 inline-block transition-transform group-open:rotate-180">
          ▾
        </span>
      </summary>
      <nav
        aria-label="カテゴリ"
        className="kicker mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-brand-navy/70"
      >
        {CATEGORIES.map((category) => (
          <Link
            key={category}
            href={`/category/${encodeURIComponent(category)}`}
            className="shrink-0 transition-colors hover:text-brand-navy"
          >
            {category}
          </Link>
        ))}
      </nav>
    </details>
  );
}
