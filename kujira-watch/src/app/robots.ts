import type { MetadataRoute } from "next";
import { RSC_DISALLOW } from "@/lib/crawlers";
import { SITE_URL } from "@/lib/site";

// 被リンク調査系のクローラーは検索にもAI引用にも寄与しないのに、36日で42,863回巡回して
// ISR再検証(=Supabase読み出し)とクローラーログを膨らませていたのでDisallowする。
const DISALLOWED_BOTS = ["AhrefsBot", "SemrushBot", "MJ12bot", "DotBot", "BLEXBot"];

// 生成AIの検索・回答・学習クローラーを明示的に許可する。`*` のAllowだけでも技術上は
// 巡回可能だが、各社の個別UAに対する運営方針を曖昧にせず、今後ルールを追加した際にも
// 意図せず巻き込まないため別グループにする。
const ALLOWED_AI_BOTS = [
  "OAI-SearchBot",
  "ChatGPT-User",
  "GPTBot",
  "ClaudeBot",
  "Claude-SearchBot",
  "Claude-User",
  "PerplexityBot",
  "Perplexity-User",
  "Google-Extended",
  "Applebot-Extended",
  "Amazonbot",
  "meta-externalagent",
];

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      { userAgent: "*", allow: "/", disallow: RSC_DISALLOW },
      ...ALLOWED_AI_BOTS.map((userAgent) => ({
        userAgent,
        allow: "/",
        disallow: RSC_DISALLOW,
      })),
      ...DISALLOWED_BOTS.map((userAgent) => ({ userAgent, disallow: "/" })),
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
