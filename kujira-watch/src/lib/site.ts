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

// 英語版（/en）用。ブランド名（日本語「大口投資家の監視ブログ」）とドメイン(kujira-watch.com)は
// 別物として管理されているため、英語ブランド名も直訳ではなく既存のクジラ文脈に合わせる。
export const SITE_NAME_EN = "Big Investor Watch";

export const SITE_DESCRIPTION_EN =
  "A blog tracking Japanese-market \"whales\" — institutional investors, insiders, and buybacks — based on EDINET large-shareholding filings.";

// 公式Xアカウント。フォロー導線（記事末尾CTA・フッター）で使用する。
export const X_SCREEN_NAME = "kujira_watch";
export const X_PROFILE_URL = `https://x.com/${X_SCREEN_NAME}`;
// フォローintent。プロフィールへの素のリンクよりワンタップ少なくフォローできる。
export const X_FOLLOW_URL = `https://x.com/intent/follow?screen_name=${X_SCREEN_NAME}`;

// 公式YouTubeチャンネル（1分ショート動画。video/publish_video.pyが平日投稿）。
// チャンネル側からサイトへはリンク済みだが、サイト側からの導線もここで持つ。
export const YOUTUBE_CHANNEL_URL = "https://www.youtube.com/@dosankoure";
