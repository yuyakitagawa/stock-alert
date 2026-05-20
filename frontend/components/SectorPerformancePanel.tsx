import type { SectorStat } from "@/lib/data";

interface Props {
  stats: SectorStat[];
}

function ReturnBar({ pct }: { pct: number }) {
  const abs = Math.min(Math.abs(pct), 30);
  const width = (abs / 30) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${pct >= 0 ? "bg-green-500" : "bg-red-500"}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={`font-mono text-sm font-bold tabular-nums ${pct >= 0 ? "text-green-400" : "text-red-400"}`}>
        {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%
      </span>
    </div>
  );
}

export default function SectorPerformancePanel({ stats }: Props) {
  if (stats.length === 0) return null;

  return (
    <section>
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="text-lg font-bold text-white">業種別成績</h2>
        <span className="text-xs text-gray-600">S買いシグナル銘柄の現在リターン（平均）</span>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase tracking-wide">
              <th className="px-4 py-2.5 text-left font-semibold">業種</th>
              <th className="px-4 py-2.5 text-right font-semibold">銘柄数</th>
              <th className="px-4 py-2.5 text-left font-semibold">平均リターン</th>
              <th className="px-4 py-2.5 text-right font-semibold">勝率</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {stats.map((s, i) => (
              <tr key={s.sector} className={i % 2 === 0 ? "bg-gray-900" : "bg-gray-900/60"}>
                <td className="px-4 py-2.5 text-gray-300 font-medium">{s.sector}</td>
                <td className="px-4 py-2.5 text-gray-500 text-right font-mono tabular-nums">{s.count}</td>
                <td className="px-4 py-2.5">
                  <ReturnBar pct={s.avgReturnPct} />
                </td>
                <td className="px-4 py-2.5 text-right">
                  <span className={`font-mono text-xs tabular-nums ${s.winRate >= 50 ? "text-green-500" : "text-orange-400"}`}>
                    {s.winRate.toFixed(0)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-gray-700 px-4 py-2 bg-gray-900/50 border-t border-gray-800">
          S買いシグナル初出現日の終値→本日終値で計算。{stats.reduce((s, r) => s + r.count, 0)} 銘柄を集計。
        </p>
      </div>
    </section>
  );
}
