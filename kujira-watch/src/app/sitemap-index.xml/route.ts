import { SITE_URL, SITEMAP_IDS } from "@/lib/site";

// generateSitemaps（app/sitemap.ts）で分割した子サイトマップ /sitemap/<id>.xml を束ねる
// sitemapindex。Next.jsはsitemapindexを自動生成しないため自前で返す。
// パスが/sitemap-index.xmlなのは、app/sitemap.xml/route.tsという名前がmetadata予約名と
// 衝突してビルドエラーになるため。従来URLの/sitemap.xml（GSC登録済み・robots.txt記載）は
// next.config.tsのrewriteでこのルートに割り当てて維持する。
// 子の一覧はデプロイ時にしか変わらないので静的でよい。
export const dynamic = "force-static";

export function GET(): Response {
  const body = [
    `<?xml version="1.0" encoding="UTF-8"?>`,
    `<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">`,
    // 子サイトマップ（urlset）と同じ1タグ1行の記述形式に揃える。
    ...SITEMAP_IDS.flatMap((id) => [
      `<sitemap>`,
      `<loc>${SITE_URL}/sitemap/${id}.xml</loc>`,
      `</sitemap>`,
    ]),
    `</sitemapindex>`,
  ].join("\n");
  return new Response(body, {
    headers: { "Content-Type": "application/xml" },
  });
}
