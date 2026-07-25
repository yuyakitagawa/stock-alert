import Link from "next/link";
import { CATEGORIES } from "@/types/article";

export default function Header() {
  return (
    <header className="border-b border-gray-200">
      <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 py-4">
        <Link href="/" className="text-xl font-bold text-gray-900">
          大口取引解説ブログ
        </Link>
        <nav className="flex flex-wrap gap-3 text-sm">
          {CATEGORIES.map((category) => (
            <Link
              key={category}
              href={`/category/${encodeURIComponent(category)}`}
              className="text-gray-600 hover:text-gray-900"
            >
              {category}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
