import type { Metadata } from "next";
import { fetchRankings, fetchSparkline, fetchSectorMap, fetchNikkeiReturn, fetchRiskRegime } from "@/lib/data";
import { fetchSimulation } from "@/lib/simulation";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import StockCard from "@/components/StockCard";
import SimulationPanel from "@/components/SimulationPanel";
import SectorPerformancePanel from "@/components/SectorPerformancePanel";
import RiskRegimeBanner from "@/components/RiskRegimeBanner";
import Link from "next/link";

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
  const [{ date, rows }, sim, sectorMap, nikkei20, risk] = await Promise.all([
    fetchRankings(),
    fetchSimulation(),
    fetchSectorMap(),
    fetchNikkeiReturn(20),
    fetchRiskRegime(),
  ]);

  // 業種別成績（本日データ）= 実際の20日リターン平均（ネットスコアではない）。
  // web_rankings の rel20（日経比）に日経20日リターンを足し戻して絶対リターンに復元。
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

  // 注目銘柄 = 「💎 買い」条件（drop<1%+net≥20+QV）の上位10銘柄
  const buyRows = rows.filter(r => r.recommend === "💎 買い");
  const featured = (buyRows.length > 0 ? buyRows : rows).slice(0, 10);
  const dateLabel = formatDate(date);

  const sparklines = await Promise.all(featured.map(r => fetchSparkline(r.code)));
  const sparklineMap = Object.fromEntries(featured.map((r, i) => [r.code, sparklines[i]]));

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar dateLabel={dateLabel} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 space-y-10">
        {/* ページ見出し（h1）＋ データ基準日（全画面で常時表示） */}
        <header className="space-y-1">
          <h1 className="text-xl sm:text-2xl font-bold text-white">日本株 AIシグナル</h1>
          <p className="text-sm text-gray-500">
            {date ? `${dateLabel} 時点のAIスコア` : "データを取得中…"}
          </p>
        </header>

        <RiskRegimeBanner risk={risk} />
        {rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
            <span className="text-5xl">📊</span>
            <p className="text-lg font-medium text-gray-500">本日のデータはまだありません</p>
            <p className="text-sm">平日16時以降に更新されます</p>
          </div>
        ) : (
          <>
            {/* Featured stocks = ネットスコア上位10銘柄 */}
            <section>
              <div className="flex items-center justify-between gap-3 mb-4">
                <h2 className="text-lg font-bold text-white whitespace-nowrap">
                  注目銘柄 <span className="text-gray-500 font-normal">Top 10</span>
                </h2>
                <Link href="/rankings" className="text-sm text-green-500 hover:text-green-400 transition-colors font-medium whitespace-nowrap">
                  全銘柄を見る →
                </Link>
              </div>
              <p className="text-xs text-gray-600 mb-4">
                {buyRows.length > 0
                  ? <>💎 買い条件（下落確率&lt;2% × ネット≥16% × 財務健全 × 株価低迷 × 業績改善）の該当 <span className="text-green-400">{buyRows.length}</span> 銘柄中 上位10件。全 {rows.length.toLocaleString()} 銘柄中。</>
                  : <>本日は💎買い条件の該当銘柄なし。ネットスコア上位10銘柄を表示。全 {rows.length.toLocaleString()} 銘柄中。</>
                }
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                {featured.map(r => (
                  <StockCard key={r.code} r={r} sparkline={sparklineMap[r.code]} />
                ))}
              </div>
            </section>

            {/* Sector performance */}
            <SectorPerformancePanel stats={sectorStats} date={dateLabel} />

            {/* Simulation */}
            <SimulationPanel positions={sim.positions} summary={sim.summary} />
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}
