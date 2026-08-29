import { displayFilerName, formatDate } from "@/lib/format";
import { getCompanyInfo } from "@/lib/companyInfo";
import { getFilerIdMap, getHoldingsByStockCode, investorPath } from "@/lib/investors";
import { getPublishedFilerNames, getPublishedStockCodes } from "@/lib/publishedPages";
import { buildRss, rssResponse } from "@/lib/rss";
import { SITE_URL } from "@/lib/site";

// 銘柄別RSS。この銘柄に提出された大量保有・変更報告書を提出者横断で新しい順に配信する。
export const revalidate = 300;

const FEED_ITEMS = 20;

export async function GET(_request: Request, { params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const [holdings, company, filerIds, publishedCodes, publishedFilers] = await Promise.all([
    getHoldingsByStockCode(code).catch(() => []),
    getCompanyInfo(code).catch(() => null),
    getFilerIdMap().catch(() => ({}) as Record<string, number>),
    getPublishedStockCodes().catch(() => new Set<string>()),
    getPublishedFilerNames().catch(() => new Set<string>()),
  ]);
  // 銘柄ページ自体を公開していない銘柄はRSSも配信しない（lib/publishedPages.ts）。
  if (!publishedCodes.has(code)) return new Response("Not Found", { status: 404 });
  const pageUrl = `${SITE_URL}/stocks/${code}`;
  const stockLabel = company?.name ? `${company.name}（${code}）` : code;

  const xml = buildRss({
    title: `${stockLabel}の大口投資家の動き`,
    link: pageUrl,
    selfUrl: `${pageUrl}/feed.xml`,
    description: `${stockLabel}に提出されたEDINET大量保有報告書・変更報告書の新着です。`,
    items: holdings
      .slice(0, FEED_ITEMS)
      // 投資家ページを公開していない提出者はリンク先が無いので、銘柄ページ自身へ向ける。
      .map((h) => {
      const label = displayFilerName(h.filerName);
      const ratio = h.holdingRatio === null ? "保有比率不明" : `保有比率${h.holdingRatio}%`;
      const prior = h.holdingRatioPrior === null ? "" : `（前回 ${h.holdingRatioPrior}%）`;
      return {
        title: `${label} ${ratio}${prior}`,
        link: publishedFilers.has(h.filerName)
          ? `${SITE_URL}${investorPath(filerIds[h.filerName], h.filerName)}`
          : pageUrl,
        guid: h.docId,
        pubDate: new Date(`${h.discDate}T00:00:00+09:00`).toUTCString(),
        description: `${formatDate(h.discDate)}にEDINETで開示。${label}の${ratio}${prior}。`,
      };
    }),
  });
  return rssResponse(xml);
}
