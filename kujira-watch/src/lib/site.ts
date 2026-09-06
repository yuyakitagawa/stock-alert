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

// Organization構造化データの alternateName。サイトは「大口投資家の監視ブログ」（ブランド名）・
// 「クジラウォッチ」（/aboutの自称）・「kujira-watch」（ドメイン）の3表記で呼ばれており、
// AI検索エンジンが別エンティティと誤認しないよう、正式名以外の呼び名をここで1つに束ねる。
// 英語名 "Big Investor Watch" は英語版のサブドメイン（en.kujira-watch.com、lib/en.ts）が
// 対応するページを持つので含める（対応ページの無い呼び名は宣言しない）。
export const SITE_ALTERNATE_NAMES = ["大口投資家の監視ブログ", "クジラウォッチ", "kujira-watch", "Big Investor Watch"];
// generateSitemapsで分割した子サイトマップのID一覧。/sitemap/<id>.xml のURLになる。
// app/sitemap.ts（子の生成）と app/sitemap.xml/route.ts（sitemapindex）の両方から参照する。
// app/sitemap.tsから直接exportしないのは、metadata routeのnamed exportは
// Next.jsのローダーがroute configとして再exportしてしまうため。
export const SITEMAP_IDS = ["pages", "stocks", "dates", "investors", "articles"] as const;
export type SitemapId = (typeof SITEMAP_IDS)[number];

// 公式Xアカウント。SNSへの導線はフッターと/about・/contactの連絡先窓口だけに絞っており、
// 本文中のフォローCTAは2026-09-06に全廃した（フォローintentの定数も同時に削除）。
export const X_SCREEN_NAME = "kujira_watch";
export const X_PROFILE_URL = `https://x.com/${X_SCREEN_NAME}`;
// Xカードの帰属表示（twitter:site / twitter:creator）用。これが無いと、サイトのURLが
// Xで共有されてもカードにアカウント名が出ず、共有のたびに得られるはずの露出を捨てている。
export const X_HANDLE = `@${X_SCREEN_NAME}`;

// 公式YouTubeチャンネル（1分ショート動画。video/publish_video.pyが平日投稿）。
// サイト側からの導線はフッターとOrganizationのsameAsだけ。
// ハンドル変更時は video/youtube_client.py の CHANNEL_URL も対で更新すること。
export const YOUTUBE_CHANNEL_URL = "https://www.youtube.com/@kujira-watch";

// Organization構造化データの sameAs。本文・フッターのリンクだけではAI検索エンジンの
// エンティティグラフに結び付かないため、公式アカウントをスキーマ側でも宣言する。
export const ORGANIZATION_SAME_AS = [X_PROFILE_URL, YOUTUBE_CHANNEL_URL];

// Organization構造化データの contactPoint。運営者は実名・メールを公開しない方針のため、
// 連絡先は公式XのみをE-E-A-T（信頼性）の連絡可能性シグナルとして宣言する。/aboutの
// 「運営者について」と同じ窓口を指す。
export const ORGANIZATION_CONTACT_POINT = {
  "@type": "ContactPoint",
  contactType: "customer support",
  url: X_PROFILE_URL,
  availableLanguage: ["ja", "en"],
};
