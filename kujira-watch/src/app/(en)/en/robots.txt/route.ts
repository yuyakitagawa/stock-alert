import { EN_SITE_URL } from "@/lib/en";

// 英語版ホストの /robots.txt。app/robots.ts（日本語版）はサイトマップとして
// kujira-watch.com/sitemap.xml を指しているので、英語版には別のrobotsを返す。
// 英語版ホストへの /robots.txt は src/proxy.ts が /en/robots.txt に rewrite してここに来る。
export const dynamic = "force-static";

export function GET(): Response {
  const body = ["User-Agent: *", "Allow: /", "", `Sitemap: ${EN_SITE_URL}/sitemap-en.xml`, ""].join("\n");
  return new Response(body, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
