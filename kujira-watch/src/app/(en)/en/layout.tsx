import type { Metadata } from "next";
import { Geist, Geist_Mono, Noto_Sans_JP } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { GoogleAnalytics } from "@next/third-parties/google";
import GaClickTracker from "@/components/GaClickTracker";
import RippleEffect from "@/components/RippleEffect";
import Header from "@/components/Header";
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

const notoSansJP = Noto_Sans_JP({
  variable: "--font-noto-sans-jp",
  subsets: ["latin"],
  weight: ["400", "500", "700", "900"],
  display: "swap",
});

const EN_SITE_URL = `${SITE_URL}/en`;

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: `${SITE_NAME_EN} | Tracking Japan's Market "Whales" from Large-Holding Filings`,
    template: `${SITE_NAME_EN} | %s`,
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
      className={`${geistSans.variable} ${geistMono.variable} ${notoSansJP.variable} h-full antialiased`}
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
        <Header locale="en" />
        <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10">
          {children}
        </main>
        <Analytics />
        <SpeedInsights />
        <GaClickTracker />
        <RippleEffect />
      </body>
      <GoogleAnalytics gaId="G-0Z3FMTXC5B" />
    </html>
  );
}
