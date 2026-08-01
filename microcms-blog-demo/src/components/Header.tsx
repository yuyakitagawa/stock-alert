import Link from "next/link";
import { CATEGORIES } from "@/types/article";

export default function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 py-4">
        <Link href="/" className="flex items-center gap-2 text-lg font-bold text-brand-navy">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-brand-blue" />
          大口取引解説ブログ
        </Link>
        <nav className="flex flex-wrap gap-2 text-sm">
          {CATEGORIES.map((category) => (
            <Link
              key={category}
              href={`/category/${encodeURIComponent(category)}`}
              className="rounded-full border border-brand-blue/40 px-3 py-1 font-medium text-brand-blue transition-colors hover:bg-brand-blue hover:text-white"
            >
              {category}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
