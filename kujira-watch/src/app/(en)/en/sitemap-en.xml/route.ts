import { getAllArticlesForSitemap, getTranslatedArticleRefs } from "@/lib/microcms";
import { EN_SITE_URL } from "@/lib/en";
import { isIndexableEnArticle, supersededArticleIds } from "@/lib/articleIndexability";

// 英語版（en.kujira-watch.com）のサイトマップ。日本語版と同じく、記事ページ側でnoindexに
// する記事（英語版の基準 isIndexableEnArticle() を満たさないもの、同一「銘柄×提出者」で
// 最新でないもの）はここにも載せない。
// ファイル名を sitemap.xml にしないのは、metadata予約名（app/sitemap.ts）とのルート衝突を
// 避けるため（日本語版が sitemap-index.xml にしているのと同じ理由）。
// microCMSの一時障害でビルドを落とさないよう、リクエスト時に生成する（取得は1時間キャッシュ）。
export const dynamic = "force-dynamic";

function urlEntry(loc: string, lastmod?: string): string[] {
  return ["<url>", `<loc>${loc}</loc>`, ...(lastmod ? [`<lastmod>${lastmod}</lastmod>`] : []), "</url>"];
}

export async function GET(): Promise<Response> {
  const [translated, allArticles] = await Promise.all([
    getTranslatedArticleRefs(),
    getAllArticlesForSitemap(),
  ]);
  // カニバリ判定は全記事（英訳の有無に関わらず）で行う。英訳済みだけでグループを作ると
  // 記事ページ側（同一銘柄の全記事で判定）と結果がずれる。
  const superseded = supersededArticleIds(allArticles);
  const indexable = translated.filter((a) => isIndexableEnArticle(a) && !superseded.has(a.id));
  const latest = indexable.map((a) => a.dealDate).sort().at(-1);

  const body = [
    `<?xml version="1.0" encoding="UTF-8"?>`,
    `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">`,
    ...urlEntry(EN_SITE_URL, latest),
    ...urlEntry(`${EN_SITE_URL}/about`),
    ...urlEntry(`${EN_SITE_URL}/privacy`),
    ...indexable.flatMap((a) => urlEntry(`${EN_SITE_URL}/articles/${a.id}`, a.dealDate)),
    `</urlset>`,
  ].join("\n");
  return new Response(body, { headers: { "Content-Type": "application/xml" } });
}
