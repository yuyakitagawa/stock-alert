"use client";
import Link from "next/link";
import type { SimPosition, SimSummary } from "@/lib/simulation";

function fmtYen(n: number) {
  return (n >= 0 ? "+" : "") + n.toLocaleString("ja-JP", { maximumFractionDigits: 0 }) + "円";
}
function fmtPct(n: number) {
  return (n >= 0 ? "+" : "") + n.toFixed(2) + "%";
}

interface Props {
  positions: SimPosition[];
  summary: SimSummary;
}

export default function SimulationPanel({ positions, summary }: Props) {
  const held = positions.filter(p => p.status === "held");
  const sold = positions.filter(p => p.status === "sold");

  const pnlColor = summary.totalPnl >= 0 ? "text-green-400" : "text-red-400";
  const pnlPctColor = summary.totalPnlPct >= 0 ? "text-green-400" : "text-red-400";

  return (
    <section className="space-y-6">
      <div className="flex items-baseline gap-3">
        <h2 className="text-lg font-bold text-white">シミュレーション</h2>
        <span className="text-xs text-gray-600">2026年1月〜 / S買い銘柄 100株ずつ</span>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="text-xs text-gray-500 mb-1">投資額（保有）</div>
          <div className="font-mono font-bold text-white text-sm">
            {summary.totalCost.toLocaleString("ja-JP")}円
          </div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="text-xs text-gray-500 mb-1">評価額</div>
          <div className="font-mono font-bold text-white text-sm">
            {summary.totalValue.toLocaleString("ja-JP")}円
          </div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="text-xs text-gray-500 mb-1">評価損益</div>
          <div className={`font-mono font-bold text-sm ${pnlColor}`}>
            {fmtYen(summary.totalPnl)}
          </div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="text-xs text-gray-500 mb-1">勝率（保有中）</div>
          <div className="font-mono font-bold text-white text-sm">
            {summary.heldCount > 0
              ? `${summary.winCount}/${summary.heldCount} (${Math.round(summary.winCount / summary.heldCount * 100)}%)`
              : "—"}
          </div>
        </div>
      </div>

      {/* Held positions */}
      {held.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-400 mb-2">保有中 ({held.length})</h3>
          <div className="overflow-x-auto rounded-xl border border-gray-800">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-gray-800 text-gray-600">
                  <th className="text-left px-3 py-2 font-medium">銘柄</th>
                  <th className="text-right px-3 py-2 font-medium">買付日</th>
                  <th className="text-right px-3 py-2 font-medium">買値</th>
                  <th className="text-right px-3 py-2 font-medium">現在値</th>
                  <th className="text-right px-3 py-2 font-medium">損益</th>
                  <th className="text-right px-3 py-2 font-medium">騰落</th>
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
                          {fmtYen(p.pnl)}
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
        <div>
          <h3 className="text-sm font-semibold text-gray-600 mb-2">売却済み ({sold.length})　※成績に含まず</h3>
          <div className="overflow-x-auto rounded-xl border border-gray-800/50">
            <table className="w-full text-xs font-mono opacity-60">
              <thead>
                <tr className="border-b border-gray-800/50 text-gray-600">
                  <th className="text-left px-3 py-2 font-medium">銘柄</th>
                  <th className="text-right px-3 py-2 font-medium">買付日</th>
                  <th className="text-right px-3 py-2 font-medium">売却日</th>
                  <th className="text-right px-3 py-2 font-medium">買値</th>
                  <th className="text-right px-3 py-2 font-medium">売値</th>
                  <th className="text-right px-3 py-2 font-medium">損益</th>
                  <th className="text-right px-3 py-2 font-medium">騰落</th>
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
                          {fmtYen(p.pnl)}
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
    </section>
  );
}
