"use client";
import type { Ranking } from "@/lib/types";
import type { RiskRegime, MarketCompare } from "@/lib/types";
import StockCard from "./StockCard";
import SectorPerformancePanel from "./SectorPerformancePanel";
import RiskRegimeBanner from "./RiskRegimeBanner";
import MarketCompareBanner from "./MarketCompareBanner";

interface SectorStat {
  sector: string;
  count: number;
  avgReturn: number;
}

interface Props {
  date: string;
  dateLabel: string;
  rows: Ranking[];
  buyRows: Ranking[];
  featured: Ranking[];
  sparklineMap: Record<string, number[] | undefined>;
  sectorStats: SectorStat[];
  risk: RiskRegime | null;
  marketCompare: MarketCompare | null;
  nikkei20: number;
}

export default function HomeContent({
  date, dateLabel, rows, buyRows, featured, sparklineMap,
  sectorStats, risk, marketCompare, nikkei20,
}: Props) {
  const nikkeiBullish = nikkei20 > 0;
  const noGems = buyRows.length === 0;

  return (
    <>
      <header className="space-y-1">
        <h1 className="text-xl sm:text-2xl font-bold text-white">日本株 AIシグナル</h1>
        <p className="text-sm text-gray-500">
          {date ? `${dateLabel} 時点のAIスコア` : "データを取得中…"}
        </p>
      </header>

      <RiskRegimeBanner risk={risk} />
      <MarketCompareBanner compare={marketCompare} />
      {rows.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
          <span className="text-5xl">📊</span>
          <p className="text-lg font-medium text-gray-500">本日のデータはまだありません</p>
          <p className="text-sm">平日16時以降に更新されます</p>
        </div>
      ) : (
        <>
          {noGems ? (
            <section>
              <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-6 py-8 text-center space-y-3">
                <p className="text-gray-400">
                  本日は💎買い条件（下落確率&lt;5% × ネット≥20 × ボラ≤30% × 90日リターン&gt;−25%）の該当銘柄はありません。
                </p>
                {nikkeiBullish && (
                  <p className="text-yellow-400 font-medium">
                    📈 日経225が上昇基調のため、個別株より日経225連動ETF（1321等）の検討を推奨します。
                  </p>
                )}
              </div>
            </section>
          ) : (
            <section>
              <div className="flex items-center justify-between gap-3 mb-4">
                <h2 className="text-lg font-bold text-white whitespace-nowrap">
                  注目銘柄 <span className="text-gray-500 font-normal">Top {featured.length}</span>
                </h2>
              </div>
              <p className="text-xs text-gray-600 mb-4">
                💎 買い条件（下落確率&lt;5% × ネット≥20 × ボラ≤30% × 90日リターン&gt;−25%）の該当 <span className="text-green-400">{buyRows.length}</span> 銘柄中 上位10件。全 {rows.length.toLocaleString()} 銘柄中。
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                {featured.map(r => (
                  <StockCard key={r.code} r={r} sparkline={sparklineMap[r.code]} />
                ))}
              </div>
            </section>
          )}

          <SectorPerformancePanel stats={sectorStats} date={dateLabel} />
        </>
      )}
    </>
  );
}
