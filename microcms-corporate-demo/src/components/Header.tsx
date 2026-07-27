"use client";

import Link from "next/link";
import { useState } from "react";

const NAV_ITEMS = [
  { label: "ガス", href: "#service-gas" },
  { label: "電気", href: "#service-electric" },
  { label: "暮らしのサービス", href: "#service-life" },
  { label: "法人のお客さま", href: "#service-business" },
  { label: "サステナビリティ", href: "#sustainability" },
  { label: "IR情報", href: "#ir" },
  { label: "ニュース", href: "/news" },
];

export default function Header() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-20 bg-white shadow-sm">
      <div className="border-b border-gray-100 bg-brand-navy text-white">
        <div className="mx-auto flex max-w-6xl items-center justify-end gap-4 px-4 py-1.5 text-xs">
          <Link href="#" className="hover:underline">
            お客さまマイページ
          </Link>
          <Link href="#" className="hover:underline">
            よくあるご質問
          </Link>
          <Link href="#contact" className="hover:underline">
            お問い合わせ
          </Link>
        </div>
      </div>
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link href="/" className="flex items-center gap-2">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-brand-blue text-sm font-bold text-white">
            ME
          </span>
          <span className="text-lg font-bold tracking-tight text-brand-navy">
            みらいエネルギー株式会社
          </span>
        </Link>
        <nav className="hidden lg:flex lg:items-center lg:gap-6">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              className="text-sm font-medium text-brand-navy transition-colors hover:text-brand-blue"
            >
              {item.label}
            </Link>
          ))}
        </nav>
        <button
          type="button"
          onClick={() => setOpen((prev) => !prev)}
          aria-expanded={open}
          aria-label="メニューを開閉"
          className="flex h-9 w-9 flex-col items-center justify-center gap-1.5 lg:hidden"
        >
          <span className="h-0.5 w-6 bg-brand-navy" />
          <span className="h-0.5 w-6 bg-brand-navy" />
          <span className="h-0.5 w-6 bg-brand-navy" />
        </button>
      </div>
      {open && (
        <nav className="flex flex-col border-t border-gray-100 px-4 py-2 lg:hidden">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              onClick={() => setOpen(false)}
              className="border-b border-gray-50 py-2.5 text-sm font-medium text-brand-navy"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
