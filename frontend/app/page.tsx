import type { Metadata } from "next";
import { fetchRankings, fetchSparkline } from "@/lib/data";
import { fetchSimulation } from "@/lib/simulation";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import StockCard from "@/components/StockCard";
import RecommendBadge from "@/components/RecommendBadge";
import SimulationPanel from "@/components/SimulationPanel";
import Link from "next/link";

export const revalidate = 0;

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
  const [{ date, rows }, sim] = await Promise.all([
    fetchRankings(),
    fetchSimulation(),
  ]);

  const sBuy    = rows.filter(r => r.recommend === "S買い");
  const aBuy    = rows.filter(r => r.recommend === "A買い");
  const caution = rows.filter(r => r.recommend === "高値警戒");
  const sell    = rows.filter(r => r.recommend === "売り検討");

  const featured = [...sBuy, ...aBuy].slice(0, 12);
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
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <SummaryCard label="S買い"   count={sBuy.length}    colorClass="text-green-400"  borderClass="border-green-900" />
                <SummaryCard label="A買い"   count={aBuy.length}    colorClass="text-green-500"  borderClass="border-green-900/50" />
                <SummaryCard label="高値警戒" count={caution.length} colorClass="text-yellow-400" borderClass="border-yellow-900/50" />
                <SummaryCard label="売り検討" count={sell.length}    colorClass="text-red-400"    borderClass="border-red-900/50" />
              </div>
            </section>

            {/* Featured stocks */}
            {featured.length > 0 && (
              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-bold text-white">注目銘柄</h2>
                  <Link href="/rankings" className="text-sm text-green-500 hover:text-green-400 transition-colors font-medium">
                    全銘柄を見る →
                  </Link>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                  {featured.map(r => (
                    <StockCard key={r.code} r={r} sparkline={sparklineMap[r.code]} />
                  ))}
                </div>
              </section>
            )}

            {/* Sell signals */}
            {sell.length > 0 && (
              <section>
                <h2 className="text-lg font-bold text-white mb-4">売り検討</h2>
                <div className="bg-red-950/20 border border-red-900/30 rounded-xl divide-y divide-gray-800/60 overflow-hidden">
                  {sell.slice(0, 6).map(r => (
                    <Link
                      key={r.code}
                      href={`/stocks/${r.code}`}
                      className="flex items-center gap-4 px-4 py-3 hover:bg-red-950/30 transition-colors"
                    >
                      <div className="flex-1 min-w-0">
                        <span className="font-semibold text-sm text-white">{r.name}</span>
                        <span className="ml-2 text-xs text-gray-600 font-mono">{r.code}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <RecommendBadge value={r.recommend} />
                        <span className="font-mono text-sm font-bold text-red-400">
                          {(r.net >= 0 ? "+" : "") + r.net.toFixed(1)}%
                        </span>
                      </div>
                    </Link>
                  ))}
                </div>
                {sell.length > 6 && (
                  <p className="text-xs text-gray-600 mt-2 text-right">
                    他 {sell.length - 6} 銘柄 →{" "}
                    <Link href="/rankings?tab=売り検討" className="text-red-500 hover:underline">全て見る</Link>
                  </p>
                )}
              </section>
            )}
          </>
        )}

        {/* Simulation */}
        <SimulationPanel positions={sim.positions} summary={sim.summary} />
      </main>

      <Footer />
    </div>
  );
}
