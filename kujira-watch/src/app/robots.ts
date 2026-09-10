import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/site";

// 被リンク調査系のクローラーは検索にもAI引用にも寄与しないのに、36日で42,863回巡回して
// ISR再検証(=Supabase読み出し)とクローラーログを膨らませていたのでDisallowする。
const DISALLOWED_BOTS = ["AhrefsBot", "SemrushBot", "MJ12bot", "DotBot", "BLEXBot"];

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      { userAgent: "*", allow: "/" },
      ...DISALLOWED_BOTS.map((userAgent) => ({ userAgent, disallow: "/" })),
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
