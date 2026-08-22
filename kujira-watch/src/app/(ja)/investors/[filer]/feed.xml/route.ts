import { displayFilerName, formatDate } from "@/lib/format";
import { getFilerHoldings, getFilerIdByName, getFilerNameById, investorPath } from "@/lib/investors";
import { buildRss, rssResponse } from "@/lib/rss";
import { SITE_URL } from "@/lib/site";

// 投資家別RSS。この提出者のEDINET大量保有・変更報告書だけを新しい順に配信する。
// 記事化されない小さな開示も含めて追えるよう、microCMSの記事ではなくEDINET開示を源にする。
export const revalidate = 300;

const FEED_ITEMS = 20;

export async function GET(_request: Request, { params }: { params: Promise<{ filer: string }> }) {
  const { filer } = await params;
  // /investors/<番号>/feed.xml が正。旧形式（提出者名）も同じ内容を返す。
  const filerId = /^\d+$/.test(filer) ? Number(filer) : null;
  const filerName =
    filerId !== null ? await getFilerNameById(filerId).catch(() => null) : decodeURIComponent(filer);
  if (!filerName) return new Response("Not Found", { status: 404 });
  const resolvedId = filerId ?? (await getFilerIdByName(filerName).catch(() => null));
  const holdings = await getFilerHoldings(filerName).catch(() => []);
  const pageUrl = `${SITE_URL}${investorPath(resolvedId, filerName)}`;
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
