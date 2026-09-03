import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { GoogleAnalytics } from "@next/third-parties/google";
import AdSenseScript from "@/components/AdSenseScript";
import GaClickTracker from "@/components/GaClickTracker";
import RippleEffect from "@/components/RippleEffect";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ThemeRegistry from "@/components/ThemeRegistry";
import { SITE_DESCRIPTION, SITE_NAME, SITE_URL, X_HANDLE, SITE_ALTERNATE_NAMES, ORGANIZATION_SAME_AS, ORGANIZATION_CONTACT_POINT } from "@/lib/site";
import "../globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// 和文はウェブフォントを読まず端末内蔵フォント（globals.cssの--font-sans）に任せる。
// next/font/googleのsubsetsは「preloadする範囲」の指定でしかなく、生成CSSからCJKの
// @font-faceは落ちない。Noto Sans JPの4ウェイトでunicode-range分割の@font-faceが
// 496個・約378KB(gzip 130KB)のレンダリングブロッキングCSSになり、さらに本文の漢字に
// 応じて70〜90KBのwoff2スライスをウェイトごとに追加ダウンロードしていた。
// AndroidのシステムフォントはNoto Sans CJK JPそのもので、iOSはHiragino Sansが載っている。
// 端末が既に持っているものを毎回落としていた形なので、見た目をほぼ変えずに丸ごと削減できる。

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: `${SITE_NAME}｜大量保有報告書から読む大口投資家の動き`,
    // 固有名（記事タイトル）を先頭にする。全記事の<title>が同じ11文字で始まっていると
    // 検索結果でどれも同じ見出しに見え、Googleにもタイトルを書き換えられやすい。
    // ブラウザタブでサイト名が切れる不利はあるが、検索結果での識別を優先する。
    template: `%s｜${SITE_NAME}`,
  },
  description: SITE_DESCRIPTION,
  alternates: {
    canonical: "/",
    types: {
      "application/rss+xml": `${SITE_URL}/feed.xml`,
    },
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
    locale: "ja_JP",
    url: SITE_URL,
    siteName: SITE_NAME,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    site: X_HANDLE,
    creator: X_HANDLE,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
};

const websiteJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  name: SITE_NAME,
  url: SITE_URL,
  description: SITE_DESCRIPTION,
  inLanguage: "ja",
};

const organizationJsonLd = {
  "@context": "https://schema.org",
  "@type": "Organization",
  name: SITE_NAME,
  alternateName: SITE_ALTERNATE_NAMES.filter((n) => n !== SITE_NAME),
  description: SITE_DESCRIPTION,
  sameAs: ORGANIZATION_SAME_AS,
  contactPoint: ORGANIZATION_CONTACT_POINT,
  url: SITE_URL,
  logo: `${SITE_URL}/logo`,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="ja"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
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
        <ThemeRegistry>
          <Header />
          <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10">
            {children}
          </main>
          <Footer />
        </ThemeRegistry>
        <AdSenseScript />
        <SpeedInsights />
        <GaClickTracker />
        <RippleEffect />
      </body>
      <GoogleAnalytics gaId="G-0Z3FMTXC5B" />
    </html>
  );
}
