import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { GoogleAnalytics } from "@next/third-parties/google";
import AdSenseScript from "@/components/AdSenseScript";
import GaClickTracker from "@/components/GaClickTracker";
import RippleEffect from "@/components/RippleEffect";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ThemeRegistry from "@/components/ThemeRegistry";
import { SITE_DESCRIPTION_EN, SITE_NAME_EN, SITE_URL } from "@/lib/site";
import "../../globals.css";

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

const EN_SITE_URL = `${SITE_URL}/en`;

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: `${SITE_NAME_EN} | Tracking Japan's Market "Whales" from Large-Holding Filings`,
    // ja版と同じ理由で固有名を先頭にする。
    template: `%s | ${SITE_NAME_EN}`,
  },
  description: SITE_DESCRIPTION_EN,
  alternates: {
    canonical: "/en",
    languages: { ja: "/", en: "/en" },
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
    card: "summary_large_image",
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
  url: EN_SITE_URL,
  logo: `${SITE_URL}/logo`,
};

export default function EnRootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-background">
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd) }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationJsonLd) }}
        />
        <ThemeRegistry>
          <Header locale="en" />
          <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10">
            {children}
          </main>
          <Footer locale="en" />
        </ThemeRegistry>
        <AdSenseScript />
        <Analytics />
        <SpeedInsights />
        <GaClickTracker />
        <RippleEffect />
      </body>
      <GoogleAnalytics gaId="G-0Z3FMTXC5B" />
    </html>
  );
}
