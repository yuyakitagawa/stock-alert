import { excerptFromHtml } from "@/lib/format";
import { getArticleList } from "@/lib/microcms";
import { SITE_DESCRIPTION, SITE_NAME, SITE_URL } from "@/lib/site";

// RSSはISR化: CDNが5分キャッシュ(s-maxage=300)し、期限切れ後は裏で再生成する。
// RSSリーダーの定期巡回が毎回microCMSまで到達するのを防ぐ。
export const revalidate = 300;

function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export async function GET() {
  const { contents } = await getArticleList({ limit: 20 });

  const items = contents
    .map((article) => {
      const url = `${SITE_URL}/articles/${article.id}`;
      const pubDate = new Date(article.publishedAt ?? article.dealDate).toUTCString();
      return `
    <item>
      <title>${escapeXml(article.title)}</title>
      <link>${url}</link>
      <guid isPermaLink="true">${url}</guid>
      <pubDate>${pubDate}</pubDate>
      <description>${escapeXml(excerptFromHtml(article.body, 200))}</description>
    </item>`;
    })
    .join("");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>${escapeXml(SITE_NAME)}</title>
    <link>${SITE_URL}</link>
    <description>${escapeXml(SITE_DESCRIPTION)}</description>
    <language>ja</language>
    <atom:link xmlns:atom="http://www.w3.org/2005/Atom" href="${SITE_URL}/feed.xml" rel="self" type="application/rss+xml" />${items}
  </channel>
</rss>`;

  return new Response(xml, {
    headers: { "Content-Type": "application/rss+xml; charset=utf-8" },
  });
}
