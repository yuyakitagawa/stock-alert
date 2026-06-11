interface SectorStat {
  sector:    string;
  count:     number;
  avgReturn: number;   // 業種内銘柄の平均20日リターン(%)（絶対値）
}

interface Props {
  stats: SectorStat[];
  date:  string;
}

export default function SectorPerformancePanel({ stats, date }: Props) {
  if (stats.length === 0) return null;

  const max = Math.max(...stats.map(s => Math.abs(s.avgReturn)), 1);

  return (
    <section>
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="text-lg font-bold text-white">業種別成績</h2>
        <span className="text-xs text-gray-600">{date} · 平均20日リターン順</span>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-3">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-1">
          {stats.map(s => (
            <div key={s.sector} className="flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-gray-800/60 transition-colors">
              <span className="text-xs text-gray-400 truncate mr-2 min-w-0">{s.sector}</span>
              <div className="flex items-center gap-1.5 shrink-0">
                <div className="w-8 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${s.avgReturn >= 0 ? "bg-green-500" : "bg-red-500"}`}
                    style={{ width: `${Math.abs(s.avgReturn) / max * 100}%` }}
                  />
                </div>
                <span className={`font-mono text-xs font-bold tabular-nums w-14 text-right ${s.avgReturn >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {s.avgReturn >= 0 ? "+" : ""}{s.avgReturn.toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
