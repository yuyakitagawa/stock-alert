import type { Metadata } from "next";
import { fetchRankings, fetchSparkline, fetchSectorMap, fetchNikkeiReturn, fetchRiskRegime } from "@/lib/data";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import HomeContent from "@/components/HomeContent";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "StockSignal — 日本株AIシグナル",
};

function formatDate(date: string) {
  if (!date) return "—";
  return new Date(date).toLocaleDateString("ja-JP", {
    year: "numeric", month: "long", day: "numeric", weekday: "short",
  });
}

export default async function HomePage() {
  const [{ date, rows }, sectorMap, nikkei20, risk] = await Promise.all([
    fetchRankings(),
    fetchSectorMap(),
    fetchNikkeiReturn(20),
    fetchRiskRegime(),
  ]);

  const sectorBuckets = new Map<string, number[]>();
  for (const r of rows) {
    if (r.rel20 == null) continue;
    const sector = sectorMap[r.code] ?? "その他";
    if (!sectorBuckets.has(sector)) sectorBuckets.set(sector, []);
    sectorBuckets.get(sector)!.push(r.rel20 + nikkei20);
  }
  const sectorStats = Array.from(sectorBuckets.entries())
    .map(([sector, rets]) => ({
      sector,
      count: rets.length,
      avgReturn: rets.reduce((s, n) => s + n, 0) / rets.length,
    }))
    .sort((a, b) => b.avgReturn - a.avgReturn);

  const buyRows = rows.filter(r => r.recommend === "💎 買い");
  const featured = (buyRows.length > 0 ? buyRows : rows).slice(0, 10);
  const dateLabel = formatDate(date);

  const sparklines = await Promise.all(featured.map(r => fetchSparkline(r.code)));
  const sparklineMap = Object.fromEntries(featured.map((r, i) => [r.code, sparklines[i]]));

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar dateLabel={dateLabel} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 space-y-10">
        <HomeContent
          date={date}
          dateLabel={dateLabel}
          rows={rows}
          buyRows={buyRows}
          featured={featured}
          sparklineMap={sparklineMap}
          sectorStats={sectorStats}
          risk={risk}
          nikkei20={nikkei20}
        />
      </main>

      <Footer />
    </div>
  );
}
