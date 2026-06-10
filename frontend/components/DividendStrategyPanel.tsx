import type { DividendCandidate } from "@/lib/types";
import Link from "next/link";

interface Props {
  candidates: DividendCandidate[];
}

export default function DividendStrategyPanel({ candidates }: Props) {
  if (!candidates.length) return null;

  return (
    <div className="bg-gray-900 rounded-xl border border-purple-800/40 p-5">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">🔄</span>
        <h2 className="text-base font-bold text-white">配当落ち後 戻し買いチャンス</h2>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        権利落ち後7〜20日の高利回り優待株。クロス解消売り一巡後の反発狙い。保有目安20営業日。6・8・9月落ちは除外済み。
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs border-b border-gray-800">
              <th className="text-left pb-2 font-medium">銘柄</th>
              <th className="text-right pb-2 font-medium">株価</th>
              <th className="text-right pb-2 font-medium">配当利回り</th>
              <th className="text-right pb-2 font-medium">ネット</th>
              <th className="text-right pb-2 font-medium">落ち後</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c) => {
              const netColor =
                c.net >= 10 ? "text-emerald-400" :
                c.net >= 5  ? "text-green-400" :
                c.net < 0   ? "text-red-400" : "text-gray-400";
              const yieldColor =
                c.div_yield >= 3 ? "text-red-400" : "text-purple-400";
              return (
                <tr key={c.code} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-2.5">
                    <Link href={`/stocks/${c.code}`} className="hover:text-blue-400 transition-colors">
                      <span className="text-white font-medium">{c.name}</span>
                      <span className="text-gray-600 text-xs ml-1.5">{c.code}</span>
                    </Link>
                  </td>
                  <td className="text-right py-2.5 text-gray-300 font-mono">
                    {c.close.toLocaleString()}円
                  </td>
                  <td className={`text-right py-2.5 font-bold font-mono ${yieldColor}`}>
                    {c.div_yield.toFixed(1)}%
                  </td>
                  <td className={`text-right py-2.5 font-bold font-mono ${netColor}`}>
                    {c.net >= 0 ? "+" : ""}{c.net.toFixed(1)}%
                  </td>
                  <td className="text-right py-2.5 text-gray-500 text-xs">
                    {c.days_since_ex}日後
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-600 mt-3">
        ※ バックテスト実績: 高利回り(3%+)×3月・12月落ち = 平均+4〜9%, 勝率75〜78%
      </p>
    </div>
  );
}
