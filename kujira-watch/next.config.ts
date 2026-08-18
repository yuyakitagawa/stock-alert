import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // /sitemap.xml はGSC登録済み・robots.txt記載の公開URL。実体はsitemapindexを返す
  // app/sitemap-index.xml/route.ts（app/sitemap.xml/route.tsはmetadata予約名と衝突するため）。
  async rewrites() {
    return [{ source: "/sitemap.xml", destination: "/sitemap-index.xml" }];
  },
  // /ranking/filings（報告書件数ランキング）は2026-08-18に廃止。2026-08-15公開で
  // インデックス済みのため404にせず、内容が最も近い/ranking/trending（開示急増投資家）へ
  // 恒久リダイレクトする。
  async redirects() {
    return [{ source: "/ranking/filings", destination: "/ranking/trending", permanent: true }];
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
