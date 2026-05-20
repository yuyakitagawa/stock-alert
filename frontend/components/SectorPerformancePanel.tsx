import type { SectorStat } from "@/lib/data";

interface Props {
  stats: SectorStat[];
}

export default function SectorPerformancePanel({ stats }: Props) {
  if (stats.length === 0) return null;

  const total = stats.reduce((s, r) => s + r.count, 0);

  return (
    <section>
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="text-lg font-bold text-white">業種別成績</h2>
        <span className="text-xs text-gray-600">S買い {total} 銘柄・平均リターン</span>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-3">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-1">
          {stats.map(s => (
            <div key={s.sector} className="flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-gray-800/60 transition-colors">
              <span className="text-xs text-gray-400 truncate mr-2 min-w-0">{s.sector}</span>
              <div className="flex items-center gap-1.5 shrink-0">
                <span className={`font-mono text-xs font-bold tabular-nums ${s.avgReturnPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {s.avgReturnPct >= 0 ? "+" : ""}{s.avgReturnPct.toFixed(1)}%
                </span>
                <span className="text-gray-700 text-xs tabular-nums">{s.count}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
