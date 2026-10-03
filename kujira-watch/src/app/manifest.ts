import type { MetadataRoute } from "next";

// 「ホーム画面に追加」/ PWAインストール用。Android Chrome は manifest が無いと
// インストール導線を出さない。アイコンは icon.tsx が生成する 192/512px を指す
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "大口投資家の監視ブログ",
    short_name: "大口投資家",
    description: "EDINET大量保有報告をもとに大口投資家の売買を毎日解説",
    start_url: "/",
    display: "standalone",
    background_color: "#16213a",
    theme_color: "#16213a",
    icons: [
      { src: "/icon/192", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon/512", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/apple-icon", sizes: "180x180", type: "image/png" },
    ],
  };
}
