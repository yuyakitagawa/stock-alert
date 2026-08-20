import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // /sitemap.xml はGSC登録済み・robots.txt記載の公開URL。実体はsitemapindexを返す
  // app/sitemap-index.xml/route.ts（app/sitemap.xml/route.tsはmetadata予約名と衝突するため）。
  async rewrites() {
    return [{ source: "/sitemap.xml", destination: "/sitemap-index.xml" }];
  },
  // 廃止したランキングURLの後始末。いずれもGSC登録済み・ヘッダー/フッターから
  // 貼られていたURLなので404にはせず、内容が最も近いページへ恒久リダイレクトする。
  // - /ranking/buys・/ranking/sells（推定金額の月間ランキング）: 2026-08-20に廃止。
  //   「いちばん大きく張った投資家」が分かるだけで銘柄選びの手掛かりにならず、常連の
  //   運用会社が並ぶだけだったため、成績で並べる/ranking/returnsへ。
  // - /ranking（旧3ヶ月リターンランキング）: 2026-08-18に廃止後は/ranking/buysへ
  //   飛ばしていたが、リダイレクトの連鎖を作らないよう/ranking/returnsへ直接飛ばす。
  // - /ranking/filings（報告書件数ランキング）: 2026-08-18に廃止。開示件数を投資家別に
  //   数えるだけで/ranking/trending（前30日比の増加件数）と重複していたため、そちらへ。
  async redirects() {
    return [
      { source: "/ranking", destination: "/ranking/returns", permanent: true },
      { source: "/ranking/buys", destination: "/ranking/returns", permanent: true },
      { source: "/ranking/sells", destination: "/ranking/returns", permanent: true },
      { source: "/ranking/filings", destination: "/ranking/trending", permanent: true },
    ];
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
