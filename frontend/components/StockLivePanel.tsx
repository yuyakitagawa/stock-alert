"use client";
import { useState, useEffect } from "react";
import type { CompanyProfile, QuarterlyEarning, DailyQuote } from "@/lib/types";

function fmtJPY(n: number | null): string {
  if (n == null) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(1)}兆円`;
  if (abs >= 1e8)  return `${(n / 1e8).toFixed(0)}億円`;
  if (abs >= 1e4)  return `${(n / 1e4).toFixed(0)}万円`;
  return `${n.toLocaleString()}円`;
}
function fmtVolume(v: number | null): string {
  if (v == null) return "—";
  if (v >= 100_000_000) return `${(v / 100_000_000).toFixed(1)}億株`;
  if (v >= 10_000)      return `${(v / 10_000).toFixed(0)}万株`;
  return `${v.toLocaleString()}株`;
}

interface Props {
  code: string;
  website?: string | null;
}

export default function StockLivePanel({ code, website }: Props) {
  const [quote, setQuote] = useState<DailyQuote | null>(null);
  const [profile, setProfile] = useState<CompanyProfile | null>(null);
  const [earnings, setEarnings] = useState<QuarterlyEarning[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const [q, p, e] = await Promise.all([
        fetch(`/api/stock/${code}/quote`).then(r => r.ok ? r.json() : null),
        fetch(`/api/stock/${code}/profile`).then(r => r.ok ? r.json() : null),
        fetch(`/api/stock/${code}/earnings`).then(r => r.ok ? r.json() : []),
      ]);
      setQuote(q);
      setProfile(p);
      setEarnings(e ?? []);
      setLoading(false);
    }
    load();
  }, [code]);

  if (loading) {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="h-28 bg-gray-900 border border-gray-800 rounded-xl" />
        <div className="h-20 bg-gray-900 border border-gray-800 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Today's market data */}
      {quote && (
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide">本日の市場データ</h2>
            {quote.date && <span className="text-xs text-gray-600 font-mono">{quote.date}</span>}
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            {quote.changePct != null && (
              <div className={`px-5 py-3 border-b border-gray-800 flex items-center gap-3 ${
                quote.changePct >= 0 ? "bg-green-950/20" : "bg-red-950/20"
              }`}>
                <span className={`font-mono text-xl font-bold ${
                  quote.changePct >= 0 ? "text-green-400" : "text-red-400"
                }`}>
                  {quote.changePct >= 0 ? "+" : ""}{quote.changePct.toFixed(2)}%
                </span>
                {quote.change != null && (
                  <span className={`font-mono text-sm ${quote.change >= 0 ? "text-green-600" : "text-red-600"}`}>
                    ({quote.change >= 0 ? "+" : ""}¥{quote.change.toLocaleString("ja-JP", { maximumFractionDigits: 0 })})
                  </span>
                )}
                {quote.prevClose != null && (
                  <span className="text-xs text-gray-600 ml-auto">
                    前日終値 ¥{quote.prevClose.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}
                  </span>
                )}
              </div>
            )}
            <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-y divide-gray-800/60">
              {[
                { label: "始値",  val: quote.open  != null ? `¥${quote.open.toLocaleString("ja-JP",  { maximumFractionDigits: 0 })}` : "—" },
                { label: "高値",  val: quote.high  != null ? `¥${quote.high.toLocaleString("ja-JP",  { maximumFractionDigits: 0 })}` : "—", color: "text-green-400" },
                { label: "安値",  val: quote.low   != null ? `¥${quote.low.toLocaleString("ja-JP",   { maximumFractionDigits: 0 })}` : "—", color: "text-red-400" },
                { label: "出来高", val: fmtVolume(quote.volume) },
              ].map(({ label, val, color }) => (
                <div key={label} className="px-4 py-3">
                  <div className="text-xs text-gray-500 mb-1">{label}</div>
                  <div className={`font-mono text-sm font-bold ${color ?? "text-white"}`}>{val}</div>
                </div>
              ))}
            </div>
            {(quote.fiftyTwoWeekLow != null || quote.fiftyTwoWeekHigh != null) && (
              <div className="px-5 py-3 border-t border-gray-800/60 flex items-center gap-4 text-xs">
                <span className="text-gray-500">52週レンジ</span>
                <span className="font-mono text-red-400">
                  安値 {quote.fiftyTwoWeekLow != null ? `¥${quote.fiftyTwoWeekLow.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}` : "—"}
                </span>
                {quote.fiftyTwoWeekLow != null && quote.fiftyTwoWeekHigh != null && quote.price != null && (
                  <div className="flex-1 flex items-center gap-1">
                    <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full"
                        style={{ width: `${Math.max(0, Math.min(100, ((quote.price - quote.fiftyTwoWeekLow) / (quote.fiftyTwoWeekHigh - quote.fiftyTwoWeekLow)) * 100))}%` }}
                      />
                    </div>
                  </div>
                )}
                <span className="font-mono text-green-400">
                  高値 {quote.fiftyTwoWeekHigh != null ? `¥${quote.fiftyTwoWeekHigh.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}` : "—"}
                </span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Company Overview */}
      {(profile?.description || website) && (
        <section className="space-y-4">
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide">会社概要</h2>
          {profile?.description && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <p className="text-gray-400 text-sm leading-relaxed">{profile.description}</p>
            </div>
          )}
          {website && (
            <div>
              <p className="text-xs text-gray-600 mb-2 font-semibold">IR</p>
              <a
                href={website}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-xs text-gray-300 hover:bg-gray-700 hover:text-white transition-colors font-medium"
              >
                {website}
                <span className="text-gray-600">↗</span>
              </a>
            </div>
          )}
        </section>
      )}

      {/* Recent Earnings */}
      {earnings.length > 0 && (
        <section>
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide mb-3">
            最新決算（直近{earnings.length}四半期）
          </h2>
          <div className="overflow-hidden rounded-xl border border-gray-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-900/80 text-gray-500 text-left border-b border-gray-800">
                  <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wide">期間</th>
                  <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-right">売上高</th>
                  <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-right">純利益</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60">
                {earnings.map((q, i) => (
                  <tr key={i} className="bg-gray-900">
                    <td className="px-4 py-2.5 font-mono text-gray-300 text-xs">{q.period}</td>
                    <td className="px-4 py-2.5 font-mono text-gray-200 text-sm text-right">{fmtJPY(q.revenue)}</td>
                    <td className={`px-4 py-2.5 font-mono text-sm font-bold text-right ${
                      q.netIncome == null ? "text-gray-500" : q.netIncome >= 0 ? "text-green-400" : "text-red-400"
                    }`}>
                      {fmtJPY(q.netIncome)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-gray-700 px-4 py-2 bg-gray-900/50 border-t border-gray-800">出典: Yahoo Finance</p>
          </div>
        </section>
      )}
    </div>
  );
}
