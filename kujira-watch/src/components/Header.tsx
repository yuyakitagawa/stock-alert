import Link from "next/link";
import HeaderMenu from "./HeaderMenu";
import StockSearch from "./StockSearch";
import VisitCounter from "./VisitCounter";
import { SITE_NAME, SITE_NAME_EN } from "@/lib/site";
import { DEAL_TYPES } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

export default function Header({ locale = "ja" }: { locale?: Locale }) {
  const homeHref = locale === "en" ? "/en" : "/";

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
              <Link href="/en" className="shrink-0 transition-colors hover:text-brand-navy">
                Top
              </Link>
              {DEAL_TYPES.map((dealType) => (
                <Link
                  key={dealType}
                  href={`/en/category/${DEAL_TYPE_EN[dealType].slug}`}
                  className="shrink-0 transition-colors hover:text-brand-navy"
                >
                  {DEAL_TYPE_EN[dealType].label}
                </Link>
              ))}
            </>
          ) : (
            <>
              <Link href="/" className="shrink-0 transition-colors hover:text-brand-navy">
                TOP
              </Link>
              <Link href="/weekly" className="shrink-0 text-brand-gold transition-colors hover:text-brand-navy">
                今週のまとめ
              </Link>
              <Link href="/ranking" className="shrink-0 transition-colors hover:text-brand-navy">
                投資家勝率ランキング
              </Link>
              <Link href="/investors" className="shrink-0 transition-colors hover:text-brand-navy">
                投資家一覧
              </Link>
              <Link href="/stocks" className="shrink-0 transition-colors hover:text-brand-navy">
                銘柄一覧
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
