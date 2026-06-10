"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useLang } from "@/contexts/LanguageContext";
import { UI } from "@/lib/i18n";

interface Props {
  dateLabel?: string;
}

export default function Navbar({ dateLabel }: Props) {
  const { lang, setLang } = useLang();
  const ui = UI[lang];
  const pathname = usePathname();

  return (
    <header className="bg-gray-950/80 backdrop-blur-sm border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 shrink-0">
          <span className="text-green-400 text-xl leading-none">&#x1F4C8;</span>
          <div>
            <div className="font-bold text-base tracking-tight text-white leading-tight">
              Stock<span className="text-green-400">Signal</span>
            </div>
            <div className="text-[9px] text-gray-600 leading-none" suppressHydrationWarning>
              {lang === "ja" ? "日本株 AI" : "Japan Stocks AI"}
            </div>
          </div>
        </Link>

        {/* Nav links */}
        <nav className="hidden sm:flex items-center gap-6 text-sm font-medium" suppressHydrationWarning>
          <Link
            href="/"
            className={`transition-colors ${pathname === "/" ? "text-white" : "text-gray-400 hover:text-white"}`}
          >
            {ui.top}
          </Link>
          <Link
            href="/rankings"
            className={`transition-colors ${pathname?.startsWith("/rankings") ? "text-white" : "text-gray-400 hover:text-white"}`}
          >
            {ui.rankings}
          </Link>
          <Link
            href="/watchlist"
            className={`transition-colors ${pathname?.startsWith("/watchlist") ? "text-white" : "text-gray-400 hover:text-white"}`}
          >
            {ui.watchlist}
          </Link>
          <Link
            href="/activity"
            className={`transition-colors ${pathname?.startsWith("/activity") ? "text-white" : "text-gray-400 hover:text-white"}`}
          >
            {ui.activity}
          </Link>
          <Link
            href="/review"
            className={`transition-colors ${pathname?.startsWith("/review") ? "text-white" : "text-gray-400 hover:text-white"}`}
          >
            {ui.review}
          </Link>
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-2">
          {dateLabel && (
            <span className="hidden md:block text-xs text-gray-500 tabular-nums">
              {dateLabel}
            </span>
          )}
          <button
            onClick={() => setLang(lang === "ja" ? "en" : "ja")}
            className="text-xs font-bold bg-gray-800 border border-gray-700 px-2.5 py-1 rounded-md hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
            suppressHydrationWarning
          >
            {lang === "ja" ? "EN" : "JP"}
          </button>
        </div>
      </div>
    </header>
  );
}
