"use client";
import Link from "next/link";
import { useLang } from "@/contexts/LanguageContext";
import { UI } from "@/lib/i18n";
import type { Ranking } from "@/lib/types";
import type { SimPosition, SimSummary } from "@/lib/simulation";
import type { RiskRegime } from "@/lib/types";
import StockCard from "./StockCard";
import SimulationPanel from "./SimulationPanel";
import SectorPerformancePanel from "./SectorPerformancePanel";
import RiskRegimeBanner from "./RiskRegimeBanner";

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
  sim: { positions: SimPosition[]; summary: SimSummary };
  risk: RiskRegime | null;
  nikkei20: number;
}

export default function HomeContent({
  date, dateLabel, rows, buyRows, featured, sparklineMap,
  sectorStats, sim, risk, nikkei20,
}: Props) {
  const { lang } = useLang();
  const ui = UI[lang];

  const nikkeiBullish = nikkei20 > 0;
  const noGems = buyRows.length === 0;

  return (
    <>
      <header className="space-y-1">
        <h1 className="text-xl sm:text-2xl font-bold text-white">{ui.pageTitle}</h1>
        <p className="text-sm text-gray-500">
          {date ? ui.aiScoreAt(dateLabel) : ui.fetching}
        </p>
      </header>

      <RiskRegimeBanner risk={risk} />
      {rows.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
          <span className="text-5xl">📊</span>
          <p className="text-lg font-medium text-gray-500">{ui.noData}</p>
          <p className="text-sm">{ui.noDataSub}</p>
        </div>
      ) : (
        <>
          <section>
            <div className="flex items-center justify-between gap-3 mb-4">
              <h2 className="text-lg font-bold text-white whitespace-nowrap">
                {ui.featured} <span className="text-gray-500 font-normal">{ui.top10}</span>
              </h2>
              <Link href="/rankings" className="text-sm text-green-500 hover:text-green-400 transition-colors font-medium whitespace-nowrap">
                {ui.viewAll}
              </Link>
            </div>
            <p className="text-xs text-gray-600 mb-4">
              {buyRows.length > 0
                ? ui.gemBuyDesc(buyRows.length, rows.length.toLocaleString())
                : noGems && nikkeiBullish
                  ? (<>
                      {ui.noGemDesc(rows.length.toLocaleString())}
                      <br />
                      <span className="text-yellow-400 font-medium">{ui.nikkeiRecommend}</span>
                    </>)
                  : ui.noGemDesc(rows.length.toLocaleString())
              }
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {featured.map(r => (
                <StockCard key={r.code} r={r} sparkline={sparklineMap[r.code]} />
              ))}
            </div>
          </section>

          <SectorPerformancePanel stats={sectorStats} date={dateLabel} />

          <SimulationPanel positions={sim.positions} summary={sim.summary} />
        </>
      )}
    </>
  );
}
