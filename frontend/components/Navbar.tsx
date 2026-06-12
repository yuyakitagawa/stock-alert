"use client";
import { useState } from "react";
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
  const [open, setOpen] = useState(false);

  const links: { href: string; label: string }[] = [
    { href: "/",          label: ui.top },
    { href: "/rankings",  label: ui.rankings },
    { href: "/watchlist", label: ui.watchlist },
    { href: "/review",    label: ui.review },
  ];

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname?.startsWith(href);

  return (
    <header className="bg-gray-950/80 backdrop-blur-sm border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 shrink-0" onClick={() => setOpen(false)}>
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

        {/* Desktop nav links */}
        <nav className="hidden sm:flex items-center gap-6 text-sm font-medium" suppressHydrationWarning>
          {links.map(l => (
            <Link
              key={l.href}
              href={l.href}
              className={`transition-colors ${isActive(l.href) ? "text-white" : "text-gray-400 hover:text-white"}`}
            >
              {l.label}
            </Link>
          ))}
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

          {/* Hamburger (mobile only) */}
          <button
            onClick={() => setOpen(o => !o)}
            aria-label={open ? "メニューを閉じる" : "メニューを開く"}
            aria-expanded={open}
            className="sm:hidden flex items-center justify-center w-9 h-9 rounded-md border border-gray-700 bg-gray-800 text-gray-300 hover:text-white hover:bg-gray-700 transition-colors"
          >
            {open ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Mobile dropdown menu */}
      {open && (
        <nav className="sm:hidden border-t border-gray-800 bg-gray-950/95 backdrop-blur-sm" suppressHydrationWarning>
          <div className="max-w-7xl mx-auto px-4 py-2 flex flex-col">
            {links.map(l => (
              <Link
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className={`py-3 px-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(l.href)
                    ? "text-white bg-gray-800/60"
                    : "text-gray-400 hover:text-white hover:bg-gray-800/40"
                }`}
              >
                {l.label}
              </Link>
            ))}
            {dateLabel && (
              <span className="px-2 py-2 text-xs text-gray-600 tabular-nums border-t border-gray-800 mt-1">
                {dateLabel}
              </span>
            )}
          </div>
        </nav>
      )}
    </header>
  );
}
