import type { Metadata } from "next";
import { fetchRankings, fetchSparkline, fetchSectorMap } from "@/lib/data";
import { fetchSimulation } from "@/lib/simulation";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import StockCard from "@/components/StockCard";
import SimulationPanel from "@/components/SimulationPanel";
import SectorPerformancePanel from "@/components/SectorPerformancePanel";
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

interface SummaryCardProps {
  label: string;
  count: number;
  colorClass: string;
  borderClass: string;
}

function SummaryCard({ label, count, colorClass, borderClass }: SummaryCardProps) {
  return (
    <div className={`bg-gray-900 rounded-xl border ${borderClass} p-5`}>
      <div className={`text-3xl font-bold font-mono ${colorClass}`}>{count}</div>
      <div className="text-sm text-gray-500 mt-1.5">{label}</div>
    </div>
  );
}

export default async function HomePage() {
  const [{ date, rows }, sim, sectorMap] = await Promise.all([
    fetchRankings(),
    fetchSimulation(),
    fetchSectorMap(),
  ]);

  // 業種別成績（本日データ）
  const sectorBuckets = new Map<string, number[]>();
  for (const r of rows) {
    const sector = sectorMap[r.code] ?? "その他";
    if (!sectorBuckets.has(sector)) sectorBuckets.set(sector, []);
    sectorBuckets.get(sector)!.push(r.net);
  }
  const sectorStats = Array.from(sectorBuckets.entries())
    .map(([sector, nets]) => ({
      sector,
      count: nets.length,
      avgNet: nets.reduce((s, n) => s + n, 0) / nets.length,
    }))
    .sort((a, b) => b.avgNet - a.avgNet);

  const sBuy    = rows.filter(r => r.recommend === "S買い");
  const aBuy    = rows.filter(r => r.recommend === "A買い");
  const neutral = rows.filter(r => r.recommend === "方向感なし");
  const weak    = rows.filter(r => r.recommend === "弱気シグナル");
  const down    = rows.filter(r => r.recommend === "下降シグナル");
  const featured = sBuy.slice(0, 24);
  const dateLabel = formatDate(date);

  const sparklines = await Promise.all(featured.map(r => fetchSparkline(r.code)));
  const sparklineMap = Object.fromEntries(featured.map((r, i) => [r.code, sparklines[i]]));

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar dateLabel={dateLabel} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 space-y-10">
        {rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
            <span className="text-5xl">📊</span>
            <p className="text-lg font-medium text-gray-500">本日のデータはまだありません</p>
            <p className="text-sm">平日16時以降に更新されます</p>
          </div>
        ) : (
          <>
            {/* Hero */}
            <section>
              <div className="flex items-baseline gap-3 mb-5">
                <h1 className="text-xl sm:text-2xl font-bold text-white">日本株 シグナル概要</h1>
                <span className="text-sm text-gray-600 font-mono">{dateLabel}</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                <SummaryCard label="S買い"      count={sBuy.length}    colorClass="text-green-400"   borderClass="border-green-900" />
                <SummaryCard label="A買い"      count={aBuy.length}    colorClass="text-green-500"   borderClass="border-green-900/50" />
                <SummaryCard label="方向感なし" count={neutral.length} colorClass="text-gray-400"    borderClass="border-gray-700" />
                <SummaryCard label="弱気"       count={weak.length}    colorClass="text-orange-400"  borderClass="border-orange-900/50" />
                <SummaryCard label="下降"       count={down.length}    colorClass="text-red-500"     borderClass="border-red-900/50" />
              </div>
              <p className="text-xs text-gray-700 text-right mt-2 font-mono">合計 {rows.length.toLocaleString()} 銘柄</p>
            </section>

            {/* Featured stocks */}
            <section>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold text-white">注目銘柄</h2>
                  <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-green-900/60 text-green-400 border border-green-800">S買い</span>
                </div>
                <Link href="/rankings" className="text-sm text-green-500 hover:text-green-400 transition-colors font-medium">
                  全銘柄を見る →
                </Link>
              </div>
              {featured.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                  {featured.map(r => (
                    <StockCard key={r.code} r={r} sparkline={sparklineMap[r.code]} />
                  ))}
                </div>
              ) : (
                <div className="bg-gray-900 border border-gray-800 rounded-xl px-6 py-8 text-center text-gray-600 text-sm">
                  現在S買いシグナルの銘柄はありません
                </div>
              )}
            </section>

          </>
        )}

        {/* Sector performance */}
        <SectorPerformancePanel stats={sectorStats} date={dateLabel} />

        {/* Simulation */}
        <SimulationPanel positions={sim.positions} summary={sim.summary} />
      </main>

      <Footer />
    </div>
  );
}
