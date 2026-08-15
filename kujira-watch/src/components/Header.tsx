"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import HeaderMenu from "./HeaderMenu";
import StockSearch from "./StockSearch";
import VisitCounter from "./VisitCounter";
import { SITE_NAME, SITE_NAME_EN } from "@/lib/site";
import { DEAL_TYPES } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

// タブが「今どのページを開いているか」を示せるよう、現在のパスと一致する
// (または配下にある)リンクだけをアクティブ表示する。
function isActiveTab(pathname: string, href: string): boolean {
  if (href === "/" || href === "/en") return pathname === href;
  return pathname === href || pathname.startsWith(`${href}/`);
}

const navLinkClass = (active: boolean) =>
  `shrink-0 border-b-2 pb-0.5 transition-colors ${
    active
      ? "border-brand-blue font-bold text-brand-blue"
      : "border-transparent hover:text-brand-navy"
  }`;

export default function Header({ locale = "ja" }: { locale?: Locale }) {
  const homeHref = locale === "en" ? "/en" : "/";
  const pathname = usePathname() ?? homeHref;

  return (
    <header className="sticky top-0 z-10 border-b border-brand-navy bg-paper/95 backdrop-blur">
      <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 pt-4 pb-2">
        <div className="flex items-center justify-between gap-2">
          <Link href={homeHref} className="group flex items-center gap-2.5">
            <span aria-hidden className="text-xl leading-none">
              🐋
            </span>
            <span className="leading-tight">
              <span className="block text-xl font-bold tracking-tight text-brand-navy sm:text-2xl">
                {locale === "en" ? SITE_NAME_EN : SITE_NAME}
              </span>
              <span className="kicker mt-0.5 hidden text-brand-blue sm:block">
                {locale === "en"
                  ? "Tracking Japan's Market \"Whales\" from Large-Holding Filings"
                  : "EDINET大量保有報告書から読む「クジラ」の動き"}
              </span>
            </span>
          </Link>
          <div className="flex shrink-0 items-center gap-3">
            <StockSearch locale={locale} />
            <VisitCounter locale={locale} />
            <HeaderMenu locale={locale} />
          </div>
        </div>
        <nav
          aria-label={locale === "en" ? "Categories" : "主要ページ"}
          className="no-scrollbar kicker flex flex-nowrap items-center gap-x-4 gap-y-1 overflow-x-auto border-t border-rule pt-2 text-brand-navy/70 sm:flex-wrap sm:overflow-visible"
        >
          {locale === "en" ? (
            <>
              <Link href="/en" aria-current={isActiveTab(pathname, "/en") ? "page" : undefined} className={navLinkClass(isActiveTab(pathname, "/en"))}>
                Top
              </Link>
              {DEAL_TYPES.map((dealType) => {
                const href = `/en/category/${DEAL_TYPE_EN[dealType].slug}`;
                return (
                  <Link
                    key={dealType}
                    href={href}
                    aria-current={isActiveTab(pathname, href) ? "page" : undefined}
                    className={navLinkClass(isActiveTab(pathname, href))}
                  >
                    {DEAL_TYPE_EN[dealType].label}
                  </Link>
                );
              })}
            </>
          ) : (
            <>
              <Link href="/" aria-current={isActiveTab(pathname, "/") ? "page" : undefined} className={navLinkClass(isActiveTab(pathname, "/"))}>
                TOP
              </Link>
              <Link
                href="/weekly"
                aria-current={isActiveTab(pathname, "/weekly") ? "page" : undefined}
                className={navLinkClass(isActiveTab(pathname, "/weekly"))}
              >
                今週のまとめ
              </Link>
              <Link
                href="/ranking"
                aria-current={isActiveTab(pathname, "/ranking") ? "page" : undefined}
                className={navLinkClass(isActiveTab(pathname, "/ranking"))}
              >
                投資家勝率ランキング
              </Link>
              <Link
                href="/investors"
                aria-current={isActiveTab(pathname, "/investors") ? "page" : undefined}
                className={navLinkClass(isActiveTab(pathname, "/investors"))}
              >
                投資家一覧
              </Link>
              <Link
                href="/stocks"
                aria-current={isActiveTab(pathname, "/stocks") ? "page" : undefined}
                className={navLinkClass(isActiveTab(pathname, "/stocks"))}
              >
                銘柄一覧
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
