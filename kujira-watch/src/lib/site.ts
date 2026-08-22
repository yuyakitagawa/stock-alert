// 独自ドメイン(kujira-watch.com)がVercelに接続済みのため、デフォルト値をそちらにしている。
// NEXT_PUBLIC_SITE_URLをVercelの環境変数で設定していればそちらが優先される。
export const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL || "https://kujira-watch.com"
).replace(/\/$/, "");

// 表示ブランド名(日本語)とドメイン(kujira-watch.com)は別物として管理する
// （ブランド名は「大口投資家の監視ブログ」、ドメインはクジラのイメージで先に確保したもの）。
export const SITE_NAME = process.env.NEXT_PUBLIC_SITE_NAME || "大口投資家の監視ブログ";

export const SITE_DESCRIPTION =
  "誰がどの日本株を買い増したか、毎日わかる。5%超の株主に開示が義務づけられた大量保有報告書（EDINET）を集計し、機関投資家・アクティビスト・自社株買いの動きを解説します。";

// 英語版（/en）用。ブランド名（日本語「大口投資家の監視ブログ」）とドメイン(kujira-watch.com)は
// 別物として管理されているため、英語ブランド名も直訳ではなく既存のクジラ文脈に合わせる。
export const SITE_NAME_EN = "Big Investor Watch";

export const SITE_DESCRIPTION_EN =
  "A blog tracking Japanese-market \"whales\" — institutional investors, insiders, and buybacks — based on EDINET large-shareholding filings.";

// generateSitemapsで分割した子サイトマップのID一覧。/sitemap/<id>.xml のURLになる。
// app/sitemap.ts（子の生成）と app/sitemap.xml/route.ts（sitemapindex）の両方から参照する。
// app/sitemap.tsから直接exportしないのは、metadata routeのnamed exportは
// Next.jsのローダーがroute configとして再exportしてしまうため。
export const SITEMAP_IDS = ["pages", "stocks", "dates", "investors", "articles", "articles-en"] as const;
export type SitemapId = (typeof SITEMAP_IDS)[number];

// 公式Xアカウント。フォロー導線（記事末尾CTA・フッター）で使用する。
export const X_SCREEN_NAME = "kujira_watch";
export const X_PROFILE_URL = `https://x.com/${X_SCREEN_NAME}`;
// フォローintent。プロフィールへの素のリンクよりワンタップ少なくフォローできる。
export const X_FOLLOW_URL = `https://x.com/intent/follow?screen_name=${X_SCREEN_NAME}`;
// Xカードの帰属表示（twitter:site / twitter:creator）用。これが無いと、サイトのURLが
// Xで共有されてもカードにアカウント名が出ず、共有のたびに得られるはずの露出を捨てている。
export const X_HANDLE = `@${X_SCREEN_NAME}`;

// 公式YouTubeチャンネル（1分ショート動画。video/publish_video.pyが平日投稿）。
// チャンネル側からサイトへはリンク済みだが、サイト側からの導線もここで持つ。
// ハンドル変更時は video/youtube_client.py の CHANNEL_URL も対で更新すること。
export const YOUTUBE_CHANNEL_URL = "https://www.youtube.com/@kujira-watch";

// 読者向けLINE公式アカウントの友だち追加URL（lin.ee/…）。アカウント未開設のため
// 通常は未設定で、未設定の間はLINE導線を一切表示しない。開設したらVercelの環境変数
// NEXT_PUBLIC_LINE_ADD_FRIEND_URL にURLを入れるだけでFollowCtaにボタンが出る。
export const LINE_ADD_FRIEND_URL = process.env.NEXT_PUBLIC_LINE_ADD_FRIEND_URL || "";
