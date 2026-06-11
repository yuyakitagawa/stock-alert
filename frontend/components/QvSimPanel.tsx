import type { QvSimTrade } from "@/lib/types";

interface Props {
  trades: QvSimTrade[];
}

function retColor(v: number | null): string {
  if (v == null) return "text-gray-500";
  return v >= 0 ? "text-green-400" : "text-red-400";
}

function fmt(v: number | null, digits = 1, prefix = ""): string {
  if (v == null) return "—";
  return `${prefix}${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export default function QvSimPanel({ trades }: Props) {
  if (trades.length === 0) {
    return (
      <section className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-6 text-center text-gray-600 text-sm">
        シミュレーションデータがありません
      </section>
    );
  }

  const closed  = trades.filter(t => t.status === "closed");
  const active  = trades.filter(t => t.status === "active");
  const returns = closed.map(t => t.return_pct).filter((v): v is number => v != null);

  const totalReturn = returns.reduce((acc, r) => {
    // 等ウェイト保有シミュレーション
    acc.push((acc[acc.length - 1] ?? 100) * (1 + r / 100));
    return acc;
  }, [] as number[]);
  const finalEquity = totalReturn[totalReturn.length - 1] ?? 100;
  const cumReturn = finalEquity - 100;
  const winRate   = returns.length > 0 ? returns.filter(r => r > 0).length / returns.length * 100 : 0;
  const avgReturn = returns.length > 0 ? returns.reduce((s, r) => s + r, 0) / returns.length : 0;
  const bigWins   = returns.filter(r => r >= 15).length;

  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-lg font-bold text-white">
          QV バックテスト
          <span className="ml-2 text-sm font-normal text-gray-500">2026-01-01 〜</span>
        </h2>
        <p className="text-xs text-gray-600">
          実際の価格データで検証。閾値: Piotroski≥6/9 × 52週安値圏45%以内 × drop≤8% × vol≤25%。最大90日保有。
        </p>
      </div>

      {/* サマリー */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {[
          { label: "累積リターン",    value: fmt(cumReturn),          color: retColor(cumReturn) },
          { label: "平均リターン",    value: fmt(avgReturn),          color: retColor(avgReturn) },
          { label: "勝率",            value: `${winRate.toFixed(0)}%`, color: "text-gray-300" },
          { label: "大勝(≥15%)",     value: `${bigWins}件`,          color: "text-gray-300" },
          { label: "トレード数",      value: `${trades.length}件`,    color: "text-gray-300" },
        ].map(item => (
          <div key={item.label} className="bg-gray-900/60 border border-gray-800 rounded-lg px-3 py-3 text-center">
            <div className={`text-lg font-bold ${item.color}`}>{item.value}</div>
            <div className="text-[10px] text-gray-600 mt-0.5">{item.label}</div>
          </div>
        ))}
      </div>

      {/* アクティブ（保有中） */}
      {active.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-bold text-yellow-500 uppercase tracking-wide">
            保有中（期末時点のまま）{active.length}件
          </h3>
          <div className="overflow-x-auto rounded-xl border border-yellow-900/40">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-gray-800 text-gray-600 text-left">
                  <th className="px-3 py-2">銘柄</th>
                  <th className="px-3 py-2 text-right">買日</th>
                  <th className="px-3 py-2 text-right">買値</th>
                  <th className="px-3 py-2 text-right">損益</th>
                  <th className="px-3 py-2 text-right">保有日</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {active.map(t => (
                  <tr key={`${t.code}-${t.entry_date}`} className="bg-yellow-950/10">
                    <td className="px-3 py-2">
                      <span className="text-white font-semibold">{t.name ?? t.code}</span>
                      <span className="text-gray-600 ml-1">{t.code}</span>
                    </td>
                    <td className="px-3 py-2 text-right text-gray-500">{t.entry_date}</td>
                    <td className="px-3 py-2 text-right text-gray-400">
                      {t.entry_price != null ? `¥${t.entry_price.toLocaleString()}` : "—"}
                    </td>
                    <td className={`px-3 py-2 text-right font-bold ${retColor(t.return_pct)}`}>
                      {fmt(t.return_pct)}
                    </td>
                    <td className="px-3 py-2 text-right text-gray-500">
                      {t.held_days != null ? `${t.held_days}日` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* クローズ済み */}
      <div className="overflow-x-auto rounded-xl border border-gray-800">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-gray-800 text-gray-600 text-left">
              <th className="px-3 py-2">銘柄</th>
              <th className="px-3 py-2 text-right">エントリー</th>
              <th className="px-3 py-2 text-right">エグジット</th>
              <th className="px-3 py-2 text-right">買値</th>
              <th className="px-3 py-2 text-right">売値</th>
              <th className="px-3 py-2 text-right">損益</th>
              <th className="px-3 py-2 text-right">保有日</th>
              <th className="px-3 py-2 text-right">理由</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {[...trades].sort((a, b) =>
              (b.entry_date ?? "").localeCompare(a.entry_date ?? "")
            ).map(t => (
              <tr key={`${t.code}-${t.entry_date}`} className="hover:bg-gray-800/30">
                <td className="px-3 py-2">
                  <span className="text-white font-semibold">{t.name ?? t.code}</span>
                  <span className="text-gray-600 ml-1">{t.code}</span>
                </td>
                <td className="px-3 py-2 text-right text-gray-500">{t.entry_date}</td>
                <td className="px-3 py-2 text-right text-gray-500">{t.exit_date ?? "—"}</td>
                <td className="px-3 py-2 text-right text-gray-400">
                  {t.entry_price != null ? `¥${t.entry_price.toLocaleString()}` : "—"}
                </td>
                <td className="px-3 py-2 text-right text-gray-400">
                  {t.exit_price != null ? `¥${t.exit_price.toLocaleString()}` : "—"}
                </td>
                <td className={`px-3 py-2 text-right font-bold ${retColor(t.return_pct)}`}>
                  {fmt(t.return_pct)}
                </td>
                <td className="px-3 py-2 text-right text-gray-500">
                  {t.held_days != null ? `${t.held_days}日` : "—"}
                </td>
                <td className="px-3 py-2 text-right text-gray-600">
                  {t.reason ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-gray-700">
        ※ 手数料・スプレッド・税金は含まない理論値。サンプル数が少ないため参考値。
      </p>
    </section>
  );
}
