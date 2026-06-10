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

  const noData = positions.length === 0;

  const pnlColor = summary.totalPnl >= 0 ? "text-green-400" : "text-red-400";
  const annColor = summary.annualizedReturnPct >= 0 ? "text-green-400" : "text-red-400";

  return (
    <section className="space-y-6">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-baseline gap-3">
          <h2 className="text-lg font-bold text-white">シミュレーション</h2>
          <span className="text-xs text-gray-600 font-mono">{summary.since}〜</span>
        </div>
        <p className="text-xs text-gray-600 leading-relaxed">
          毎日のネットスコア上位10銘柄を100株ずつ購入し、ネットスコアが5%未満に下がった日に売却した場合の仮想成績です。
          手数料・税金は含みません。
        </p>
      </div>

      {noData && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-6 text-center text-gray-600 text-sm">
          データが蓄積されると実績が表示されます
        </div>
      )}

      {!noData && (
      <>
        {/* Annualized return hero */}
        <div className="bg-gradient-to-r from-green-950/60 to-gray-900/60 border border-green-800/40 rounded-xl p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <span className="text-xs font-semibold text-green-500 tracking-wide uppercase">年率換算リターン</span>
                <span className="text-xs text-gray-600" title="全シグナルを順次複利で再投資した場合の年率換算。観測期間のリターンを1年間にスケールした参考値です。">(?)</span>
              </div>
              <div className={`text-5xl font-bold font-mono ${annColor}`}>
                {summary.annualizedReturnPct >= 0 ? "+" : ""}{summary.annualizedReturnPct.toFixed(1)}
                <span className="text-2xl text-gray-500">%</span>
              </div>
              <div className="text-xs text-gray-500 mt-1.5">
                期間リターン {summary.compoundReturnPct >= 0 ? "+" : ""}{summary.compoundReturnPct.toFixed(1)}%
                <span className="text-gray-700 ml-2">（損益合計÷総投資額）を {summary.since} 〜 今日まで年率換算</span>
              </div>
            </div>
            <div className="text-right space-y-2 shrink-0 pt-1">
              <div className="text-xs text-gray-500">
                <span className="text-gray-300 font-bold text-base">{summary.allCount}</span>
                <span className="ml-1">シグナル</span>
              </div>
              <div className="text-xs text-gray-500">
                勝率 <span className="text-white font-bold">
                  {summary.allCount > 0 ? Math.round(summary.allWinCount / summary.allCount * 100) : 0}%
                </span>
              </div>
              <div className="text-xs text-gray-500">
                平均 <span className={`font-bold ${summary.avgReturnPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {summary.avgReturnPct >= 0 ? "+" : ""}{summary.avgReturnPct.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Accuracy summary */}
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500 font-semibold tracking-wide uppercase">シグナル精度</span>
            <span className="text-xs text-gray-700">全 {summary.allCount} シグナル（保有中 {summary.heldCount} + 売却済 {summary.soldCount}）</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {/* Win rate */}
            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <span className="text-xs text-gray-500">勝率</span>
                <span className="text-xs text-gray-700" title="買付時より現在値または売値が高い銘柄の割合">(?)</span>
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
              <div className="text-xs text-gray-600">{summary.allWinCount} 勝 / {summary.allCount - summary.allWinCount} 負</div>
            </div>

            {/* Avg return */}
            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <span className="text-xs text-gray-500">平均騰落率</span>
                <span className="text-xs text-gray-700" title="全ポジションの騰落率の単純平均">(?)</span>
              </div>
              <div className={`text-2xl font-bold font-mono ${summary.avgReturnPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                {summary.avgReturnPct >= 0 ? "+" : ""}{summary.avgReturnPct.toFixed(2)}
                <span className="text-base text-gray-500">%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full" />
              <div className="text-xs text-gray-600">買いシグナル後の平均</div>
            </div>

            {/* Max gain */}
            <div className="space-y-2">
              <div className="text-xs text-gray-500">最大利益</div>
              <div className="text-2xl font-bold font-mono text-green-400">
                +{summary.maxGainPct.toFixed(2)}
                <span className="text-base text-gray-500">%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-green-800 rounded-full w-full" />
              </div>
              <div className="text-xs text-gray-600">ベストパフォーマー</div>
            </div>

            {/* Max loss */}
            <div className="space-y-2">
              <div className="text-xs text-gray-500">最大損失</div>
              <div className="text-2xl font-bold font-mono text-red-400">
                {summary.maxLossPct.toFixed(2)}
                <span className="text-base text-gray-500">%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-red-900 rounded-full w-full" />
              </div>
              <div className="text-xs text-gray-600">ワーストパフォーマー</div>
            </div>
          </div>
        </div>

        {/* Portfolio summary */}
        <div className="space-y-2">
          <p className="text-xs text-gray-600">全シグナルの損益合計（保有中の含み損益 + 売却済みの確定損益）</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">総投資額</div>
              <div className="font-mono font-bold text-white text-sm">
                {summary.totalCost.toLocaleString("ja-JP")}円
              </div>
              <div className="text-xs text-gray-700 mt-1">全買付価格 × 100株の合計</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">損益合計</div>
              <div className={`font-mono font-bold text-sm ${pnlColor}`}>
                {fmtYen(summary.totalPnl)}
              </div>
              <div className="text-xs text-gray-700 mt-1">確定 + 含み損益の合計</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">単純平均リターン</div>
              <div className={`font-mono font-bold text-sm ${pnlColor}`}>
                {fmtPct(summary.totalPnlPct)}
              </div>
              <div className="text-xs text-gray-700 mt-1">損益合計 ÷ 総投資額（参考）</div>
            </div>
          </div>
        </div>

        {/* Held positions */}
        {held.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-baseline gap-2">
              <h3 className="text-sm font-semibold text-gray-400">保有中 ({held.length}銘柄)</h3>
              <span className="text-xs text-gray-700">買いシグナル日に買付、まだ売りシグナルが出ていない</span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-gray-800">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-600">
                    <th className="text-left px-3 py-2 font-medium">銘柄</th>
                    <th className="text-right px-3 py-2 font-medium">買付日</th>
                    <th className="text-right px-3 py-2 font-medium">買値</th>
                    <th className="text-right px-3 py-2 font-medium">現在値</th>
                    <th className="text-right px-3 py-2 font-medium">損益 (100株)</th>
                    <th className="text-right px-3 py-2 font-medium">騰落率</th>
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
          <div className="space-y-2">
            <div className="flex items-baseline gap-2">
              <h3 className="text-sm font-semibold text-gray-600">売却済み ({sold.length}銘柄)</h3>
              <span className="text-xs text-gray-700">下降シグナルが出た日に売却したと仮定</span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-gray-800/50">
              <table className="w-full text-xs font-mono opacity-60">
                <thead>
                  <tr className="border-b border-gray-800/50 text-gray-600">
                    <th className="text-left px-3 py-2 font-medium">銘柄</th>
                    <th className="text-right px-3 py-2 font-medium">買付日</th>
                    <th className="text-right px-3 py-2 font-medium">売却日</th>
                    <th className="text-right px-3 py-2 font-medium">買値</th>
                    <th className="text-right px-3 py-2 font-medium">売値</th>
                    <th className="text-right px-3 py-2 font-medium">損益 (100株)</th>
                    <th className="text-right px-3 py-2 font-medium">騰落率</th>
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

        {/* Disclaimer */}
        <p className="text-xs text-gray-700 border-t border-gray-800/60 pt-4">
          ※ 本シミュレーションは参考情報です。実際の投資判断はご自身の責任で行ってください。
          年率換算は「損益合計÷総投資額」を観測期間から年率換算したもので、将来の成果を保証するものではありません。
          売買タイミングはシグナル発生日の終値を使用。手数料・スリッページ・税金は考慮していません。
        </p>
      </>)}
    </section>
  );
}
