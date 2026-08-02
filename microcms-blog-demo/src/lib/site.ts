// ドメイン・ブランド名が未確定のため環境変数を優先し、未設定時は現行のVercelドメインにフォールバックする。
// 独自ドメイン確定後は Vercel の環境変数を書き換えるだけで全ページに反映される。
export const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL || "https://stock-alert-lyart.vercel.app"
).replace(/\/$/, "");

export const SITE_NAME = process.env.NEXT_PUBLIC_SITE_NAME || "大口投資家の監視ブログ";

export const SITE_DESCRIPTION =
  "EDINET大量保有報告書をもとに、機関投資家・インサイダー・自社株買いなど大口投資家の動きを監視・解説するブログです。";
