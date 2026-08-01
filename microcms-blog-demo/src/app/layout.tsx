import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Header from "@/components/Header";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "大口取引解説ブログ（microCMS検証用ダミーサイト）",
  description: "機関投資家買い・インサイダー買い・自社株買いなど、株式市場の大口取引を解説するダミーサイトです。",
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
      <body className="flex min-h-full flex-col bg-section-tint">
        <Header />
        <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8">
          {children}
        </main>
        <footer className="border-t border-gray-200 bg-brand-navy py-6 text-center text-xs text-white/70">
          microCMS検証用ダミーサイト・本番運用は想定していません
        </footer>
      </body>
    </html>
  );
}
