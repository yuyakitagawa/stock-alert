import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // /sitemap.xml はGSC登録済み・robots.txt記載の公開URL。実体はsitemapindexを返す
  // app/sitemap-index.xml/route.ts（app/sitemap.xml/route.tsはmetadata予約名と衝突するため）。
  async rewrites() {
    return [{ source: "/sitemap.xml", destination: "/sitemap-index.xml" }];
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
