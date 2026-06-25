import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "StockSignal — 日本株AIシグナル",
    template: "%s | StockSignal",
  },
  description: "AIが毎日算出する日本株のネットスコア（上昇確率 − 下落確率）。注目銘柄をネットスコア順で確認。",
  keywords: ["日本株", "AIシグナル", "株式投資", "ネットスコア", "株スクリーニング"],
  openGraph: {
    type: "website",
    locale: "ja_JP",
    siteName: "StockSignal",
    title: "StockSignal — 日本株AIシグナル",
    description: "AIが毎日算出する日本株のネットスコア（上昇確率 − 下落確率）。注目銘柄をネットスコア順で確認。",
  },
  twitter: {
    card: "summary",
    title: "StockSignal — 日本株AIシグナル",
    description: "AIが毎日算出する日本株のネットスコア。注目銘柄をネットスコア順で確認。",
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
        {children}
      </body>
    </html>
  );
}
