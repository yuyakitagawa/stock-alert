import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import { GoogleAnalytics } from "@next/third-parties/google";
import { SITE_URL, X_HANDLE, X_PROFILE_URL, SITE_ALTERNATE_NAMES, ORGANIZATION_SAME_AS, ORGANIZATION_CONTACT_POINT } from "@/lib/site";
import { EN_SITE_URL, SITE_DESCRIPTION_EN, SITE_NAME_EN } from "@/lib/en";
import "../../globals.css";

// 英語版（en.kujira-watch.com）のルートレイアウト。公開URLにはパスの /en が付かない
// （src/proxy.ts がホスト名で rewrite する）ので、内部リンクは /articles/... の形で書く。
//
// 日本語側の Header / Footer / ThemeRegistry（MUI）は使わない: 文言が日本語固定で、
// locale 引数を戻すと変更範囲がコンポーネント全面に広がる（2026-08-29に外した経緯）。
// 英語版の目的は「英語面にクローラーが来るか」の実測なので、装飾は最小限にする。
// AdSense は入れない（審査・薄いコンテンツ判定に英語面を巻き込まない）。GA4 だけ
// 同じプロパティで取り、hostName で日英を分ける。

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(EN_SITE_URL),
  title: {
    default: `${SITE_NAME_EN} | Tracking Japan's Market "Whales" from Large-Holding Filings`,
    template: `%s | ${SITE_NAME_EN}`,
  },
  description: SITE_DESCRIPTION_EN,
  alternates: {
    canonical: "/",
    languages: { ja: SITE_URL, en: EN_SITE_URL },
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    alternateLocale: "ja_JP",
    url: EN_SITE_URL,
    siteName: SITE_NAME_EN,
    title: SITE_NAME_EN,
    description: SITE_DESCRIPTION_EN,
  },
  twitter: {
    card: "summary",
    site: X_HANDLE,
    creator: X_HANDLE,
    title: SITE_NAME_EN,
    description: SITE_DESCRIPTION_EN,
  },
};

const websiteJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  name: SITE_NAME_EN,
  url: EN_SITE_URL,
  description: SITE_DESCRIPTION_EN,
  inLanguage: "en",
};

const organizationJsonLd = {
  "@context": "https://schema.org",
  "@type": "Organization",
  name: SITE_NAME_EN,
  alternateName: SITE_ALTERNATE_NAMES.filter((n) => n !== SITE_NAME_EN),
  description: SITE_DESCRIPTION_EN,
  sameAs: ORGANIZATION_SAME_AS,
  contactPoint: ORGANIZATION_CONTACT_POINT,
  url: EN_SITE_URL,
  logo: `${SITE_URL}/logo`,
};

const NAV_LINKS = [
  { href: "/", label: "Latest" },
  { href: "/about", label: "About" },
];

export default function EnRootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col bg-background">
        <script
          type="application/ld+json"
          // JSON.stringify of a static, code-defined object - no user input, safe to inline.
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd) }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationJsonLd) }}
        />
        <header className="sticky top-0 z-10 border-b border-brand-navy bg-paper/95 backdrop-blur-sm">
          <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-4 px-4 py-3">
            <Link href="/" className="flex items-center gap-2 text-brand-navy no-underline">
              <span aria-hidden className="text-2xl">🐋</span>
              <span className="text-lg font-bold leading-tight">{SITE_NAME_EN}</span>
            </Link>
            <nav aria-label="Main" className="flex items-center gap-4 text-sm font-medium">
              {NAV_LINKS.map((link) => (
                <Link key={link.href} href={link.href} className="text-brand-navy hover:text-brand-blue">
                  {link.label}
                </Link>
              ))}
              <a href={SITE_URL} className="text-ink-tertiary hover:text-brand-blue" hrefLang="ja" lang="ja">
                日本語
              </a>
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10">{children}</main>
        <footer className="border-t border-rule bg-paper">
          <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-6 text-xs text-ink-tertiary">
            <p className="m-0">
              © {SITE_NAME_EN}. Data: EDINET large-shareholding reports (Japan FSA). Not investment advice.
            </p>
            <nav aria-label="Footer" className="flex flex-wrap gap-4">
              <Link href="/about" className="hover:text-brand-blue">About &amp; Disclaimer</Link>
              <Link href="/privacy" className="hover:text-brand-blue">Privacy</Link>
              <a href={X_PROFILE_URL} target="_blank" rel="noopener noreferrer" className="hover:text-brand-blue">
                {X_HANDLE} on X
              </a>
              <a href={SITE_URL} hrefLang="ja" lang="ja" className="hover:text-brand-blue">
                日本語版
              </a>
            </nav>
          </div>
        </footer>
      </body>
      <GoogleAnalytics gaId="G-0Z3FMTXC5B" />
    </html>
  );
}
