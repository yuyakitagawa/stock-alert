import type { Metadata } from "next";
import { Geist, Geist_Mono, Noto_Sans_JP } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { GoogleAnalytics } from "@next/third-parties/google";
import GaClickTracker from "@/components/GaClickTracker";
import Header from "@/components/Header";
import ThemeRegistry from "@/components/ThemeRegistry";
import { SITE_DESCRIPTION, SITE_NAME, SITE_URL } from "@/lib/site";
import "../globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const notoSansJP = Noto_Sans_JP({
  variable: "--font-noto-sans-jp",
  subsets: ["latin"],
  weight: ["400", "500", "700", "900"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: `${SITE_NAME}｜大量保有報告書から読む「クジラ」の動き`,
    // サイト名を先頭にする（記事タイトルが長いとブラウザタブで末尾が切れ、
    // サイト名が末尾だと何のサイトか分からなくなるため）。
    template: `${SITE_NAME}｜%s`,
  },
  description: SITE_DESCRIPTION,
  alternates: {
    canonical: "/",
    languages: { ja: "/", en: "/en" },
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
    alternateLocale: "en_US",
    url: SITE_URL,
    siteName: SITE_NAME,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
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
      className={`${geistSans.variable} ${geistMono.variable} ${notoSansJP.variable} h-full antialiased`}
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
        </ThemeRegistry>
        <Analytics />
        <SpeedInsights />
        <GaClickTracker />
      </body>
      <GoogleAnalytics gaId="G-0Z3FMTXC5B" />
    </html>
  );
}
