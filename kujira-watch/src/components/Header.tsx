import Link from "next/link";
import { SITE_NAME } from "@/lib/site";
import { CATEGORIES } from "@/types/article";

export default function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 py-4">
        <Link href="/" className="flex items-center gap-2 text-lg font-bold text-brand-navy">
          <span
            aria-hidden
            className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-brand-blue text-base"
          >
            🐋
          </span>
          {SITE_NAME}
        </Link>
        <nav className="flex flex-wrap items-center gap-2 text-sm">
          {CATEGORIES.map((category) => (
            <Link
              key={category}
              href={`/category/${encodeURIComponent(category)}`}
              className="rounded-full border border-brand-blue/40 px-3 py-1 font-medium text-brand-blue transition-colors hover:bg-brand-blue hover:text-white"
            >
              {category}
            </Link>
          ))}
          <Link
            href="/about"
            className="ml-auto text-xs font-medium text-gray-500 hover:text-brand-blue"
          >
            運営者情報
          </Link>
        </nav>
      </div>
    </header>
  );
}
