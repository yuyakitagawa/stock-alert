import Link from "next/link";
import HeaderMenu from "./HeaderMenu";
import { SITE_NAME } from "@/lib/site";
import { CATEGORIES } from "@/types/article";

export default function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-brand-navy bg-paper/95 backdrop-blur">
      <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 pt-4 pb-2">
        <div className="flex items-center justify-between gap-2">
          <Link href="/" className="group flex items-center gap-2.5">
            <span aria-hidden className="text-xl leading-none">
              🐋
            </span>
            <span className="leading-tight">
              <span className="block text-xl font-bold tracking-tight text-brand-navy sm:text-2xl">
                {SITE_NAME}
              </span>
              <span className="kicker mt-0.5 hidden text-brand-blue sm:block">
                EDINET大量保有報告書から読む「クジラ」の動き
              </span>
            </span>
          </Link>
          <HeaderMenu />
        </div>
        {/* モバイルでは折り返さず1行の横スクロールにして、フィルターが縦に何行も
            積み重なってページ本文を押し下げないようにする（sm以上では通常の折り返し）。 */}
        <nav
          aria-label="カテゴリ"
          className="no-scrollbar kicker flex flex-nowrap items-center gap-x-4 gap-y-1 overflow-x-auto border-t border-rule pt-2 text-brand-navy/70 sm:flex-wrap sm:overflow-visible"
        >
          <Link
            href="/weekly"
            className="shrink-0 text-brand-gold transition-colors hover:text-brand-navy"
          >
            今週の動き
          </Link>
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
      </div>
    </header>
  );
}
