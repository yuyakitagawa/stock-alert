// 独自ドメイン(kujira-watch.com)はVercel側のDNS/ドメイン追加が完了するまでは未接続のため、
// NEXT_PUBLIC_SITE_URLが未設定の間は現行のVercelドメインにフォールバックする。
// Vercelでドメイン接続とNEXT_PUBLIC_SITE_URL="https://kujira-watch.com"の設定が完了したら、
// このデフォルト値もkujira-watch.comに切り替える。
export const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL || "https://stock-alert-lyart.vercel.app"
).replace(/\/$/, "");

export const SITE_NAME = process.env.NEXT_PUBLIC_SITE_NAME || "クジラウォッチ";

export const SITE_DESCRIPTION =
  "EDINET大量保有報告書をもとに、機関投資家・インサイダー・自社株買いなど「クジラ(大口投資家)」の動きを監視・解説するブログです。";
