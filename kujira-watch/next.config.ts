import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // /sitemap.xml はGSC登録済み・robots.txt記載の公開URL。実体はsitemapindexを返す
  // app/sitemap-index.xml/route.ts（app/sitemap.xml/route.tsはmetadata予約名と衝突するため）。
  async rewrites() {
    return [{ source: "/sitemap.xml", destination: "/sitemap-index.xml" }];
  },
  // /ranking は「3ヶ月リターンランキング」だったが、推定損益の算出が誤っていたため
  // 2026-08-18に廃止（tools/filer_win_rate.py・filer_win_rateテーブルごと削除）。
  // GSC登録済み・ヘッダー/フッターから貼られていたURLなので、404にせず
  // タブの先頭になった買い増しランキングへ恒久リダイレクトする。
  async redirects() {
    return [{ source: "/ranking", destination: "/ranking/buys", permanent: true }];
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "images.microcms-assets.io",
      },
    ],
    // 最適化済み画像のCDNキャッシュTTL(既定4時間)。microCMSは画像を差し替えると
    // URL自体が変わる(実質immutable)ため、31日まで延ばして再最適化コストを削る。
    minimumCacheTTL: 2678400,
  },
};

export default nextConfig;
