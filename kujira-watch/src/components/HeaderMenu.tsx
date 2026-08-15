"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SITE_NAME, SITE_NAME_EN } from "@/lib/site";
import { DEAL_TYPE_EN, EN_SLUG_TO_DEAL_TYPE } from "@/lib/dealTypeInfo";
import type { DealType } from "@/types/article";
import { UI, type Locale } from "@/lib/i18n";

type MenuLink = {
  href: string;
  label: string;
};

// 現在のパスから、もう一方のロケールの等価URLを計算する。
// 記事(id)・銘柄(code)はロケール間で共通のキーなのでprefixの付け替えだけで対応できるが、
// カテゴリだけは日本語文字列⇄英語slugの変換が要る。/weekly, /investors, /date, /faq は
// 英語版が無いためホームにフォールバックする。
function alternatePath(pathname: string): string {
  const enArticle = pathname.match(/^\/en\/articles\/(.+)$/);
  if (enArticle) return `/articles/${enArticle[1]}`;
  const jaArticle = pathname.match(/^\/articles\/(.+)$/);
  if (jaArticle) return `/en/articles/${jaArticle[1]}`;

  const enStock = pathname.match(/^\/en\/stocks\/(.+)$/);
  if (enStock) return `/stocks/${enStock[1]}`;
  const jaStock = pathname.match(/^\/stocks\/(.+)$/);
  if (jaStock) return `/en/stocks/${jaStock[1]}`;

  const enCategory = pathname.match(/^\/en\/category\/([^/]+)$/);
  if (enCategory) {
    const dealType = EN_SLUG_TO_DEAL_TYPE[enCategory[1]];
    return dealType ? `/category/${encodeURIComponent(dealType)}` : "/";
  }
  const jaCategory = pathname.match(/^\/category\/([^/]+)$/);
  if (jaCategory) {
    const dealType = decodeURIComponent(jaCategory[1]) as DealType;
    const info = DEAL_TYPE_EN[dealType];
    return info ? `/en/category/${info.slug}` : "/en";
  }

  if (pathname === "/en/about") return "/about";
  if (pathname === "/about") return "/en/about";

  return pathname.startsWith("/en") ? "/" : "/en";
}

// オートスクロールで記事一覧が際限なく伸びるため、ページ最下部のフッターまで
// スクロールしてたどり着くのが実質困難になった。運営者情報・免責事項等は
// ヘッダー右上のハンバーガーメニューに集約し、常にアクセスできるようにする。
export default function HeaderMenu({ locale = "ja" }: { locale?: Locale }) {
  const [open, setOpen] = useState(false);
  const year = new Date().getFullYear();
  const pathname = usePathname();
  const t = UI[locale];

  const close = () => setOpen(false);

  const siteLinks: MenuLink[] =
    locale === "en"
      ? [
          { href: "/en", label: "Top" },
          { href: "/en/about", label: t.aboutMenuLabel },
        ]
      : [
          { href: "/", label: "TOP" },
          { href: "/weekly", label: "今週のまとめ" },
          { href: "/ranking", label: "投資家勝率ランキング" },
          { href: "/investors", label: "投資家一覧" },
          { href: "/stocks", label: "銘柄一覧" },
          { href: "/about", label: "運営者情報・免責事項" },
          { href: "/faq", label: "よくある質問" },
          { href: "/feed.xml", label: "RSSフィード" },
        ];

  const currentPath = pathname ?? (locale === "en" ? "/en" : "/");
  const otherHref = alternatePath(currentPath);
  const jaHref = locale === "ja" ? currentPath : otherHref;
  const enHref = locale === "en" ? currentPath : otherHref;

  return (
    // PCでは中央寄せの本文カラムではなく、画面の本当の右端にハンバーガーを固定する
    // (ヘッダーはposition:stickyでabsolute配置の基準になる)
    <div className="shrink-0 sm:absolute sm:right-4 sm:top-4">
      <button
        type="button"
        aria-label={t.menuLabel}
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
        aria-label={locale === "en" ? "Close menu" : "メニューを閉じる"}
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
        aria-label={t.menuLabel}
        className={`fixed right-0 top-0 z-50 h-dvh w-[86%] max-w-xs transform overflow-y-auto border-l border-rule bg-paper shadow-2xl transition-transform duration-300 ease-out ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between border-b border-rule px-4 py-3">
          <span className="flex items-center gap-2 text-sm font-bold text-brand-navy">
            <span aria-hidden className="text-base leading-none">
              🐋
            </span>
            {locale === "en" ? SITE_NAME_EN : SITE_NAME}
          </span>
          <button
            type="button"
            aria-label={locale === "en" ? "Close menu" : "メニューを閉じる"}
            onClick={close}
            className="flex h-8 w-8 items-center justify-center text-brand-navy hover:text-brand-gold"
          >
            <span aria-hidden className="text-lg leading-none">
              ✕
            </span>
          </button>
        </div>

        <p className="px-4 py-3 text-xs leading-relaxed text-foreground/70">
          {locale === "en" ? (
            <>
              {SITE_NAME_EN} is a blog that tracks who is moving which stock, when, and how much — based on public information such as EDINET large-shareholding reports (Japan&rsquo;s &ldquo;5% rule&rdquo; filings) — covering institutional investors, activist funds, insiders, and share buybacks (collectively &ldquo;whales&rdquo;: large investors with enough capital to move the market).
            </>
          ) : (
            <>
              {SITE_NAME}は、EDINETの大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
              アクティビストファンド・インサイダー・自社株買いといった「クジラ」（相場を動かすほどの
              資金力を持つ大口投資家の俗称）が、どの銘柄をいつ・どれくらいの規模で動かしたかを
              日次でまとめて解説するブログです。
            </>
          )}
        </p>

        <p className="kicker bg-section-tint px-4 py-1.5 text-brand-navy/60">{t.langSectionLabel}</p>
        <nav aria-label={t.langSectionLabel} className="border-b border-rule">
          <Link
            href={jaHref}
            onClick={close}
            aria-current={locale === "ja" ? "page" : undefined}
            className={`flex items-center justify-between border-t border-rule/60 px-4 py-3.5 text-sm first:border-t-0 hover:bg-section-tint ${
              locale === "ja" ? "font-bold text-brand-blue" : "text-brand-navy"
            }`}
          >
            日本語
            <span aria-hidden className="text-foreground/30">
              ›
            </span>
          </Link>
          <Link
            href={enHref}
            onClick={close}
            aria-current={locale === "en" ? "page" : undefined}
            className={`flex items-center justify-between border-t border-rule/60 px-4 py-3.5 text-sm hover:bg-section-tint ${
              locale === "en" ? "font-bold text-brand-blue" : "text-brand-navy"
            }`}
          >
            English
            <span aria-hidden className="text-foreground/30">
              ›
            </span>
          </Link>
        </nav>

        <p className="kicker bg-section-tint px-4 py-1.5 text-brand-navy/60">
          {locale === "en" ? "Menu" : "メニュー"}
        </p>
        <nav aria-label={locale === "en" ? "Menu" : "メニュー"} className="border-b border-rule">
          {siteLinks.map((link) => (
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
          {locale === "en"
            ? "This site is commentary based on public EDINET large-shareholding filings and is not investment advice."
            : "本サイトはEDINET大量保有報告書等の公開情報をもとにした解説であり、投資助言ではありません。"}
        </p>
        <p className="mt-2 px-4 pb-4 text-foreground/40">
          © {year} {locale === "en" ? SITE_NAME_EN : SITE_NAME}
        </p>
      </div>
    </div>
  );
}
