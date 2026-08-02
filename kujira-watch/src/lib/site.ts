// 独自ドメイン(kujira-watch.com)がVercelに接続済みのため、デフォルト値をそちらにしている。
// NEXT_PUBLIC_SITE_URLをVercelの環境変数で設定していればそちらが優先される。
export const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL || "https://kujira-watch.com"
).replace(/\/$/, "");

// 表示ブランド名(日本語)とドメイン(kujira-watch.com)は別物として管理する
// （ブランド名は「大口投資家の監視ブログ」、ドメインはクジラのイメージで先に確保したもの）。
export const SITE_NAME = process.env.NEXT_PUBLIC_SITE_NAME || "大口投資家の監視ブログ";

export const SITE_DESCRIPTION =
  "EDINET大量保有報告書をもとに、機関投資家・インサイダー・自社株買いなど「クジラ(大口投資家)」の動きを監視・解説するブログです。";
