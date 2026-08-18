import { SITE_NAME } from "./site";

// 投資家別・銘柄別のRSS（/investors/[filer]/feed.xml・/stocks/[code]/feed.xml）と
// サイト全体のRSS（/feed.xml）で共用するXML組み立て。
// アカウント機能を持たない代わりに「この投資家・この銘柄の新しい開示だけ追う」手段として
// RSSを用意する（2026-08-19のユーザーインタビューで最も要望の多かった再訪動線）。

export function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export type RssItem = {
  title: string;
  link: string;
  guid?: string;
  pubDate: string;
  description: string;
};

export function buildRss(channel: {
  title: string;
  link: string;
  selfUrl: string;
  description: string;
  items: RssItem[];
}): string {
  const items = channel.items
    .map(
      (item) => `
    <item>
      <title>${escapeXml(item.title)}</title>
      <link>${item.link}</link>
      <guid isPermaLink="${item.guid ? "false" : "true"}">${escapeXml(item.guid ?? item.link)}</guid>
      <pubDate>${item.pubDate}</pubDate>
      <description>${escapeXml(item.description)}</description>
    </item>`
    )
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>${escapeXml(`${channel.title}｜${SITE_NAME}`)}</title>
    <link>${channel.link}</link>
    <description>${escapeXml(channel.description)}</description>
    <language>ja</language>
    <atom:link xmlns:atom="http://www.w3.org/2005/Atom" href="${channel.selfUrl}" rel="self" type="application/rss+xml" />${items}
  </channel>
</rss>`;
}

export function rssResponse(xml: string): Response {
  return new Response(xml, {
    headers: { "Content-Type": "application/rss+xml; charset=utf-8" },
  });
}
