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
  //   数えるだけだったため当時は/ranking/trendingへ飛ばしていたが、その/ranking/trendingも
  //   廃止したのでリダイレクトの連鎖を作らないよう/ranking/returnsへ直接飛ばす。
  // - /ranking/trending（開示急増投資家ランキング）: 2026-08-21に廃止。開示件数の「増加分」順は
  //   変更報告書を大量に出す常連が上位に来るだけで、買いか売りか・対象銘柄が分からず
  //   次の行動につながらなかった。投資家軸のランキングは成績で並べる/ranking/returnsに一本化。
  // 英語版(/en)は2026-08-29に廃止。GSC実測でも/en配下は検索流入の山が無く
  // （直近14日: EN記事1,046本にブラウザPV1,544・最大7PV、日本語版は最大73PV）、
  // 英訳のために記事1本あたり約3割多い出力トークンを払い続ける形になっていた。
  // 既にインデックス済みのURLがあるので404にはせず、対応する日本語ページへ恒久リダイレクトする。
  async redirects() {
    return [
      { source: "/en", destination: "/", permanent: true },
      { source: "/en/articles/:id", destination: "/articles/:id", permanent: true },
      { source: "/en/stocks/:code", destination: "/stocks/:code", permanent: true },
      { source: "/en/about", destination: "/about", permanent: true },
      { source: "/en/privacy", destination: "/privacy", permanent: true },
      { source: "/en/investors", destination: "/investors", permanent: true },
      // 英語カテゴリはslug（activist等）で、日本語カテゴリ名との対応表は廃止済み。
      // 個別に振り分けず、分類の一覧が並ぶトップへ寄せる。
      { source: "/en/category/:slug", destination: "/", permanent: true },
      { source: "/en/:path*", destination: "/", permanent: true },
      { source: "/ranking", destination: "/ranking/returns", permanent: true },
      { source: "/ranking/buys", destination: "/ranking/returns", permanent: true },
      { source: "/ranking/sells", destination: "/ranking/returns", permanent: true },
      { source: "/ranking/filings", destination: "/ranking/returns", permanent: true },
      { source: "/ranking/trending", destination: "/ranking/returns", permanent: true },
    ];
  },
  images: {
    // Vercelの画像最適化ではなくmicroCMSの画像API(imgix)でリサイズ・WebP変換する。
    // 2026-08-27にVercelの最適化枠を使い切り、本番の全アイキャッチがHTTP 402
    // (OPTIMIZED_IMAGE_REQUEST_PAYMENT_REQUIRED)になって表示できなくなったため。
    // 記事は毎日増え、画像を差し替えるたびにURLが変わって最適化がやり直しになるので、
    // 枠のある仕組みに載せ続けること自体が持たない。microCMS側の変換は転送量
    // (Hobbyで20GB/月)にしか効かない。remotePatterns/minimumCacheTTLはVercelの
    // 最適化にしか効かない設定なので、カスタムローダーへの切り替えと同時に外した。
    loader: "custom",
    loaderFile: "./src/lib/imageLoader.ts",
  },
};

export default nextConfig;
