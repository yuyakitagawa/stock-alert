import { displayFilerName, formatDate } from "@/lib/format";
import { getCompanyInfo } from "@/lib/companyInfo";
import { getHoldingsByStockCode } from "@/lib/investors";
import { buildRss, rssResponse } from "@/lib/rss";
import { SITE_URL } from "@/lib/site";

// 銘柄別RSS。この銘柄に提出された大量保有・変更報告書を提出者横断で新しい順に配信する。
export const revalidate = 300;

const FEED_ITEMS = 20;

export async function GET(_request: Request, { params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const [holdings, company] = await Promise.all([
    getHoldingsByStockCode(code).catch(() => []),
    getCompanyInfo(code).catch(() => null),
  ]);
  const pageUrl = `${SITE_URL}/stocks/${code}`;
  const stockLabel = company?.name ? `${company.name}（${code}）` : code;

  const xml = buildRss({
    title: `${stockLabel}の大口投資家の動き`,
    link: pageUrl,
    selfUrl: `${pageUrl}/feed.xml`,
    description: `${stockLabel}に提出されたEDINET大量保有報告書・変更報告書の新着です。`,
    items: holdings.slice(0, FEED_ITEMS).map((h) => {
      const label = displayFilerName(h.filerName);
      const ratio = h.holdingRatio === null ? "保有比率不明" : `保有比率${h.holdingRatio}%`;
      const prior = h.holdingRatioPrior === null ? "" : `（前回 ${h.holdingRatioPrior}%）`;
      return {
        title: `${label} ${ratio}${prior}`,
        link: `${SITE_URL}/investors/${encodeURIComponent(h.filerName)}`,
        guid: h.docId,
        pubDate: new Date(`${h.discDate}T00:00:00+09:00`).toUTCString(),
        description: `${formatDate(h.discDate)}にEDINETで開示。${label}の${ratio}${prior}。`,
      };
    }),
  });
  return rssResponse(xml);
}
