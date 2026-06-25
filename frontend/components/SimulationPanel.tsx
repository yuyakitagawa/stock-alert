"use client";
import Link from "next/link";
import type { SimPosition, SimSummary } from "@/lib/simulation";
import { useLang } from "@/contexts/LanguageContext";
import { UI } from "@/lib/i18n";

function fmtMoney(n: number, suffix: string) {
  return (n >= 0 ? "+" : "") + n.toLocaleString("ja-JP", { maximumFractionDigits: 0 }) + suffix;
}
function fmtPct(n: number) {
  return (n >= 0 ? "+" : "") + n.toFixed(2) + "%";
}

interface Props {
  positions: SimPosition[];
  summary: SimSummary;
}

export default function SimulationPanel({ positions, summary }: Props) {
  const { lang } = useLang();
  const ui = UI[lang];

  const held = positions.filter(p => p.status === "held");
  const sold = positions.filter(p => p.status === "sold");
  const noData = positions.length === 0;
  const pnlColor = summary.totalPnl >= 0 ? "text-green-400" : "text-red-400";
  const annColor = summary.annualizedReturnPct >= 0 ? "text-green-400" : "text-red-400";

  return (
    <section className="space-y-6">
      <div className="space-y-1">
        <div className="flex items-baseline gap-3">
          <h2 className="text-lg font-bold text-white">{ui.simTitle}</h2>
          <span className="text-xs text-gray-600 font-mono">{summary.since}〜</span>
        </div>
        <p className="text-xs text-gray-600 leading-relaxed">{ui.simDesc}</p>
      </div>

      {noData && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-6 text-center text-gray-600 text-sm">
          {ui.simNoData}
        </div>
      )}

      {!noData && (
      <>
        {/* Annualized return hero */}
        <div className="bg-gradient-to-r from-green-950/60 to-gray-900/60 border border-green-800/40 rounded-xl p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <span className="text-xs font-semibold text-green-500 tracking-wide uppercase">{ui.simAnnualized}</span>
                <span className="text-xs text-gray-600" title={ui.simAnnualizedTip}>(?)</span>
              </div>
              <div className={`text-5xl font-bold font-mono ${annColor}`}>
                {summary.annualizedReturnPct >= 0 ? "+" : ""}{summary.annualizedReturnPct.toFixed(1)}
                <span className="text-2xl text-gray-500">%</span>
              </div>
              <div className="text-xs text-gray-500 mt-1.5">
                {ui.simPeriodReturn} {summary.compoundReturnPct >= 0 ? "+" : ""}{summary.compoundReturnPct.toFixed(1)}%
                <span className="text-gray-600 ml-2">{ui.simPeriodReturnDesc(summary.since)}</span>
              </div>
            </div>
            <div className="text-right space-y-2 shrink-0 pt-1">
              <div className="text-xs text-gray-500">
                <span className="text-gray-300 font-bold text-base">{summary.allCount}</span>
                <span className="ml-1">{ui.simSignals}</span>
              </div>
              <div className="text-xs text-gray-500">
                {ui.simWinRate} <span className="text-white font-bold">
                  {summary.allCount > 0 ? Math.round(summary.allWinCount / summary.allCount * 100) : 0}%
                </span>
              </div>
              <div className="text-xs text-gray-500">
                {ui.simAvg} <span className={`font-bold ${summary.avgReturnPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {summary.avgReturnPct >= 0 ? "+" : ""}{summary.avgReturnPct.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Accuracy summary */}
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500 font-semibold tracking-wide uppercase">{ui.simAccuracy}</span>
            <span className="text-xs text-gray-600">{ui.simAccuracySub(summary.allCount, summary.heldCount, summary.soldCount)}</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <span className="text-xs text-gray-500">{ui.simWinRate}</span>
                <span className="text-xs text-gray-600" title={ui.simWinRateTip}>(?)</span>
              </div>
              <div className="text-2xl font-bold font-mono text-white">
                {summary.allCount > 0 ? Math.round(summary.allWinCount / summary.allCount * 100) : 0}
                <span className="text-base text-gray-500">%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full transition-all"
                  style={{ width: `${summary.allCount > 0 ? Math.round(summary.allWinCount / summary.allCount * 100) : 0}%` }}
                />
              </div>
              <div className="text-xs text-gray-600">{summary.allWinCount} {ui.simWin} / {summary.allCount - summary.allWinCount} {ui.simLoss}</div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <span className="text-xs text-gray-500">{ui.simAvgChange}</span>
                <span className="text-xs text-gray-600" title={ui.simAvgChangeTip}>(?)</span>
              </div>
              <div className={`text-2xl font-bold font-mono ${summary.avgReturnPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                {summary.avgReturnPct >= 0 ? "+" : ""}{summary.avgReturnPct.toFixed(2)}
                <span className="text-base text-gray-500">%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full" />
              <div className="text-xs text-gray-600">{ui.simAvgAfterSignal}</div>
            </div>

            <div className="space-y-2">
              <div className="text-xs text-gray-500">{ui.simMaxGain}</div>
              <div className="text-2xl font-bold font-mono text-green-400">
                +{summary.maxGainPct.toFixed(2)}
                <span className="text-base text-gray-500">%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-green-800 rounded-full w-full" />
              </div>
              <div className="text-xs text-gray-600">{ui.simBestPerformer}</div>
            </div>

            <div className="space-y-2">
              <div className="text-xs text-gray-500">{ui.simMaxLoss}</div>
              <div className="text-2xl font-bold font-mono text-red-400">
                {summary.maxLossPct.toFixed(2)}
                <span className="text-base text-gray-500">%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-red-900 rounded-full w-full" />
              </div>
              <div className="text-xs text-gray-600">{ui.simWorstPerformer}</div>
            </div>
          </div>
        </div>

        {/* Portfolio summary */}
        <div className="space-y-2">
          <p className="text-xs text-gray-600">{ui.simPortfolioDesc}</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">{ui.simTotalInvest}</div>
              <div className="font-mono font-bold text-white text-sm">
                {summary.totalCost.toLocaleString("ja-JP")}{ui.yen}
              </div>
              <div className="text-xs text-gray-600 mt-1">{ui.simTotalInvestDesc}</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">{ui.simTotalPnl}</div>
              <div className={`font-mono font-bold text-sm ${pnlColor}`}>
                {fmtMoney(summary.totalPnl, ui.yen)}
              </div>
              <div className="text-xs text-gray-600 mt-1">{ui.simTotalPnlDesc}</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">{ui.simSimpleAvgReturn}</div>
              <div className={`font-mono font-bold text-sm ${pnlColor}`}>
                {fmtPct(summary.totalPnlPct)}
              </div>
              <div className="text-xs text-gray-600 mt-1">{ui.simSimpleAvgReturnDesc}</div>
            </div>
          </div>
        </div>

        {/* Held positions */}
        {held.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-baseline gap-2">
              <h3 className="text-sm font-semibold text-gray-400">{ui.simHeld(held.length)}</h3>
              <span className="text-xs text-gray-600">{ui.simHeldDesc}</span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-gray-800">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-600">
                    <th className="text-left px-3 py-2 font-medium">{ui.simStock}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simBuyDate}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simBuyPrice}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simCurrentPrice}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simPnl100}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simChangePct}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/50">
                  {held
                    .sort((a, b) => b.pnlPct - a.pnlPct)
                    .map(p => {
                      const up = p.pnl >= 0;
                      return (
                        <tr key={p.code} className="hover:bg-gray-800/40 transition-colors">
                          <td className="px-3 py-2">
                            <Link href={`/stocks/${p.code}`} className="hover:text-green-400 transition-colors">
                              <span className="text-white font-semibold">{p.name}</span>
                              <span className="text-gray-600 ml-1.5">{p.code}</span>
                            </Link>
                          </td>
                          <td className="text-right px-3 py-2 text-gray-500">{p.buyDate}</td>
                          <td className="text-right px-3 py-2 text-gray-400">{p.buyPrice.toLocaleString()}</td>
                          <td className="text-right px-3 py-2 text-gray-300">{p.currentPrice?.toLocaleString() ?? "—"}</td>
                          <td className={`text-right px-3 py-2 font-bold ${up ? "text-green-400" : "text-red-400"}`}>
                            {fmtMoney(p.pnl, ui.yen)}
                          </td>
                          <td className={`text-right px-3 py-2 font-bold ${up ? "text-green-400" : "text-red-400"}`}>
                            {fmtPct(p.pnlPct)}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Sold positions */}
        {sold.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-baseline gap-2">
              <h3 className="text-sm font-semibold text-gray-600">{ui.simSold(sold.length)}</h3>
              <span className="text-xs text-gray-600">{ui.simSoldDesc}</span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-gray-800/50">
              <table className="w-full text-xs font-mono opacity-60">
                <thead>
                  <tr className="border-b border-gray-800/50 text-gray-600">
                    <th className="text-left px-3 py-2 font-medium">{ui.simStock}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simBuyDate}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simSellDate}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simBuyPrice}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simSellPrice}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simPnl100}</th>
                    <th className="text-right px-3 py-2 font-medium">{ui.simChangePct}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/30">
                  {sold
                    .sort((a, b) => (b.sellDate ?? "").localeCompare(a.sellDate ?? ""))
                    .map(p => {
                      const up = p.pnl >= 0;
                      return (
                        <tr key={p.code}>
                          <td className="px-3 py-2">
                            <Link href={`/stocks/${p.code}`} className="hover:opacity-100 transition-opacity">
                              <span className="text-gray-400">{p.name}</span>
                              <span className="text-gray-600 ml-1.5">{p.code}</span>
                            </Link>
                          </td>
                          <td className="text-right px-3 py-2 text-gray-600">{p.buyDate}</td>
                          <td className="text-right px-3 py-2 text-gray-600">{p.sellDate}</td>
                          <td className="text-right px-3 py-2 text-gray-600">{p.buyPrice.toLocaleString()}</td>
                          <td className="text-right px-3 py-2 text-gray-600">{p.sellPrice?.toLocaleString() ?? "—"}</td>
                          <td className={`text-right px-3 py-2 ${up ? "text-green-600" : "text-red-600"}`}>
                            {fmtMoney(p.pnl, ui.yen)}
                          </td>
                          <td className={`text-right px-3 py-2 ${up ? "text-green-600" : "text-red-600"}`}>
                            {fmtPct(p.pnlPct)}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <p className="text-xs text-gray-600 border-t border-gray-800/60 pt-4">
          {ui.simDisclaimer}
        </p>
      </>)}
    </section>
  );
}
