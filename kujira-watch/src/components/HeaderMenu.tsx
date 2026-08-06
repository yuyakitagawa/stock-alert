"use client";

import { useState } from "react";
import Link from "next/link";
import { SITE_NAME } from "@/lib/site";

type MenuLink = {
  href: string;
  label: string;
};

const SITE_LINKS: MenuLink[] = [
  { href: "/about", label: "運営者情報・免責事項" },
  { href: "/faq", label: "よくある質問" },
  { href: "/feed.xml", label: "RSSフィード" },
];

// オートスクロールで記事一覧が際限なく伸びるため、ページ最下部のフッターまで
// スクロールしてたどり着くのが実質困難になった。運営者情報・免責事項等は
// ヘッダー右上のハンバーガーメニューに集約し、常にアクセスできるようにする。
export default function HeaderMenu() {
  const [open, setOpen] = useState(false);
  const year = new Date().getFullYear();

  const close = () => setOpen(false);

  return (
    <div className="shrink-0">
      <button
        type="button"
        aria-label="メニュー"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="flex h-9 w-9 items-center justify-center text-brand-navy hover:text-brand-gold"
      >
        <span aria-hidden className="text-xl leading-none">
          ☰
        </span>
      </button>

      {/* 背景の暗幕。タップで閉じる */}
      <button
        type="button"
        aria-hidden={!open}
        aria-label="メニューを閉じる"
        onClick={close}
        tabIndex={open ? 0 : -1}
        className={`fixed inset-0 z-40 bg-black/40 transition-opacity duration-300 ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      />

      {/* 右からスライドインするフルハイトのメニューパネル */}
      <div
        role="dialog"
        aria-modal="true"
        aria-label="メニュー"
        className={`fixed right-0 top-0 z-50 h-dvh w-[86%] max-w-xs transform overflow-y-auto border-l border-rule bg-paper shadow-2xl transition-transform duration-300 ease-out ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between border-b border-rule px-4 py-3">
          <span className="flex items-center gap-2 text-sm font-bold text-brand-navy">
            <span aria-hidden className="text-base leading-none">
              🐋
            </span>
            {SITE_NAME}
          </span>
          <button
            type="button"
            aria-label="メニューを閉じる"
            onClick={close}
            className="flex h-8 w-8 items-center justify-center text-brand-navy hover:text-brand-gold"
          >
            <span aria-hidden className="text-lg leading-none">
              ✕
            </span>
          </button>
        </div>

        <p className="px-4 py-3 text-xs leading-relaxed text-foreground/70">
          {SITE_NAME}は、EDINETの大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
          アクティビストファンド・インサイダー・自社株買いといった「クジラ」（相場を動かすほどの
          資金力を持つ大口投資家の俗称）が、どの銘柄をいつ・どれくらいの規模で動かしたかを
          日次でまとめて解説するブログです。
        </p>

        <p className="kicker bg-section-tint px-4 py-1.5 text-brand-navy/60">サイトについて</p>
        <nav aria-label="サイトについて" className="border-b border-rule">
          {SITE_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={close}
              className="flex items-center justify-between border-t border-rule/60 px-4 py-3.5 text-sm text-brand-navy first:border-t-0 hover:bg-section-tint"
            >
              {link.label}
              <span aria-hidden className="text-foreground/30">
                ›
              </span>
            </Link>
          ))}
        </nav>

        <p className="mt-3 px-4 text-[11px] leading-relaxed text-foreground/70">
          本サイトはEDINET大量保有報告書等の公開情報をもとにした解説であり、投資助言ではありません。
        </p>
        <p className="mt-2 px-4 pb-4 text-foreground/40">
          © {year} {SITE_NAME}
        </p>
      </div>
    </div>
  );
}
