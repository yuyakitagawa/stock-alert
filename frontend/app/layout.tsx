import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StockSignal — 日本株AIシグナル",
  description: "AIが選ぶ日本株の買い・売りシグナルを毎日配信",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#111827" />
      </head>
      <body>{children}</body>
    </html>
  );
}
