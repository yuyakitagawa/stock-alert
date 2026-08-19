import { displayFilerName, formatDate } from "@/lib/format";
import { getFilerHoldings } from "@/lib/investors";
import { buildRss, rssResponse } from "@/lib/rss";
import { SITE_URL } from "@/lib/site";

// 投資家別RSS。この提出者のEDINET大量保有・変更報告書だけを新しい順に配信する。
// 記事化されない小さな開示も含めて追えるよう、microCMSの記事ではなくEDINET開示を源にする。
export const revalidate = 300;

const FEED_ITEMS = 20;

export async function GET(_request: Request, { params }: { params: Promise<{ filer: string }> }) {
  const { filer } = await params;
  const filerName = decodeURIComponent(filer);
  const holdings = await getFilerHoldings(filerName).catch(() => []);
  const pageUrl = `${SITE_URL}/investors/${encodeURIComponent(filerName)}`;
  const label = displayFilerName(filerName);

  const xml = buildRss({
    title: `${label}の開示`,
    link: pageUrl,
    selfUrl: `${pageUrl}/feed.xml`,
    description: `${label}がEDINETへ提出した大量保有報告書・変更報告書の新着です。`,
    items: holdings.slice(0, FEED_ITEMS).map((h) => {
      const ratio = h.holdingRatio === null ? "保有比率不明" : `保有比率${h.holdingRatio}%`;
      const prior = h.holdingRatioPrior === null ? "" : `（前回 ${h.holdingRatioPrior}%）`;
      return {
        title: `${h.issuerName}（${h.issuerCode}）${ratio}${prior}`,
        link: `${SITE_URL}/stocks/${h.issuerCode}`,
        guid: h.docId,
        pubDate: new Date(`${h.discDate}T00:00:00+09:00`).toUTCString(),
        description: `${formatDate(h.discDate)}にEDINETで開示。${label}の${h.issuerName}（${h.issuerCode}）に対する${ratio}${prior}。`,
      };
    }),
  });
  return rssResponse(xml);
}
