import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "@/contexts/LanguageContext";

export const metadata: Metadata = {
  title: {
    default: "StockSignal — 日本株AIシグナル",
    template: "%s | StockSignal",
  },
  description: "AIが毎日算出する日本株の買い・売りシグナル。S買い・A買いシグナルをリアルタイムで確認。",
  keywords: ["日本株", "AIシグナル", "株式投資", "買いシグナル", "株スクリーニング"],
  openGraph: {
    type: "website",
    locale: "ja_JP",
    siteName: "StockSignal",
    title: "StockSignal — 日本株AIシグナル",
    description: "AIが毎日算出する日本株の買い・売りシグナル。S買い・A買いシグナルをリアルタイムで確認。",
  },
  twitter: {
    card: "summary",
    title: "StockSignal — 日本株AIシグナル",
    description: "AIが毎日算出する日本株の買い・売りシグナル。",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja" className="dark">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#0a0f1a" />
      </head>
      <body className="min-h-screen antialiased">
        <LanguageProvider>{children}</LanguageProvider>
      </body>
    </html>
  );
}
