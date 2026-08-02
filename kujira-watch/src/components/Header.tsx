import Link from "next/link";
import { SITE_NAME } from "@/lib/site";
import { CATEGORIES } from "@/types/article";

export default function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-3xl flex-col gap-2 px-4 py-3">
        <Link href="/" className="flex items-center gap-2">
          <span
            aria-hidden
            className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-blue text-base"
          >
            🐋
          </span>
          <span className="leading-tight">
            <span className="block text-lg font-bold text-brand-navy">{SITE_NAME}</span>
            <span className="block text-xs text-gray-500">
              EDINET大量保有報告書から読む「クジラ」の動き
            </span>
          </span>
        </Link>
        {/* モバイルでは折り返さず1行の横スクロールにして、フィルターが縦に何行も
            積み重なってページ本文を押し下げないようにする（sm以上では通常の折り返し）。 */}
        <nav
          aria-label="カテゴリ"
          className="no-scrollbar flex flex-nowrap items-center gap-2 overflow-x-auto text-sm sm:flex-wrap sm:overflow-visible"
        >
          {CATEGORIES.map((category) => (
            <Link
              key={category}
              href={`/category/${encodeURIComponent(category)}`}
              className="shrink-0 rounded-full border border-brand-blue/40 px-3 py-1 font-medium text-brand-blue transition-colors hover:bg-brand-blue hover:text-white"
            >
              {category}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
