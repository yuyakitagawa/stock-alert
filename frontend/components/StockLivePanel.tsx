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
  name: string;
  sector?: string | null;
}

export default function StockLivePanel({ code, name, sector }: Props) {
  const [quote, setQuote]           = useState<DailyQuote | null>(null);
  const [profile, setProfile]       = useState<CompanyProfile | null>(null);
  const [earnings, setEarnings]     = useState<QuarterlyEarning[]>([]);
  const [description, setDescription] = useState<string | null>(null);
  const [loading, setLoading]       = useState(true);
  const [descLoading, setDescLoading] = useState(true);
  const [expanded, setExpanded]     = useState(false);

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

    async function loadDescription() {
      // localStorage キャッシュ確認（7日間有効）
      const cacheKey = `desc:${code}`;
      try {
        const cached = localStorage.getItem(cacheKey);
        if (cached) {
          const { text, ts } = JSON.parse(cached);
          if (Date.now() - ts < 7 * 86400 * 1000) {
            setDescription(text);
            setDescLoading(false);
            return;
          }
        }
      } catch { /* ignore */ }

      const params = new URLSearchParams({ name, sector: sector ?? "" });
      const res = await fetch(`/api/stock/${code}/description?${params}`);
      const data = res.ok ? await res.json() : { description: null };
      const text = data.description ?? null;
      setDescription(text);
      setDescLoading(false);
      if (text) {
        try { localStorage.setItem(cacheKey, JSON.stringify({ text, ts: Date.now() })); } catch { /* ignore */ }
      }
    }

    load();
    loadDescription();
  }, [code, name, sector]);

  const website  = profile?.website ?? null;
  const employees = profile?.employees ?? null;
  const TRUNCATE    = 200;

  return (
    <div className="space-y-8">

      {/* ── 会社説明 ──────────────────────────────────────── */}
      <section className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">この会社について</h2>
          {(sector || employees) && (
            <div className="flex items-center gap-3 text-xs text-gray-600">
              {sector && <span className="bg-gray-800 px-2 py-0.5 rounded-full">{sector}</span>}
              {employees && <span>{employees.toLocaleString()} 人</span>}
            </div>
          )}
        </div>

        {(loading || descLoading) ? (
          <div className="space-y-2 animate-pulse">
            <div className="h-3 bg-gray-800 rounded w-full" />
            <div className="h-3 bg-gray-800 rounded w-5/6" />
            <div className="h-3 bg-gray-800 rounded w-4/6" />
          </div>
        ) : description ? (
          <div>
            <p className="text-sm text-gray-300 leading-relaxed">
              {expanded || description.length <= TRUNCATE
                ? description
                : description.slice(0, TRUNCATE) + "…"}
            </p>
            {description.length > TRUNCATE && (
              <button
                onClick={() => setExpanded(v => !v)}
                className="text-xs text-green-500 hover:text-green-400 mt-1.5 transition-colors"
              >
                {expanded ? "折りたたむ" : "続きを読む"}
              </button>
            )}
          </div>
        ) : (
          <p className="text-sm text-gray-600">概要情報を取得できませんでした</p>
        )}

        {website && (
          <a
            href={website}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            公式サイト ↗
          </a>
        )}
      </section>

      {/* ── 本日の市場データ ─────────────────────────────── */}
      {(loading || quote) && (
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide">本日の市場データ</h2>
            {quote?.date && <span className="text-xs text-gray-600 font-mono">{quote.date}</span>}
          </div>

          {loading ? (
            <div className="h-32 bg-gray-900 border border-gray-800 rounded-xl animate-pulse" />
          ) : quote ? (
            <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
              {quote.changePct != null && (
                <div className={`px-5 py-3 border-b border-gray-800 flex items-center gap-3 ${
                  quote.changePct >= 0 ? "bg-green-950/20" : "bg-red-950/20"
                }`}>
                  <span className={`font-mono text-xl font-bold ${quote.changePct >= 0 ? "text-green-400" : "text-red-400"}`}>
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
                  { label: "始値",   val: quote.open   != null ? `¥${quote.open.toLocaleString("ja-JP",  { maximumFractionDigits: 0 })}` : "—" },
                  { label: "高値",   val: quote.high   != null ? `¥${quote.high.toLocaleString("ja-JP",  { maximumFractionDigits: 0 })}` : "—", color: "text-green-400" },
                  { label: "安値",   val: quote.low    != null ? `¥${quote.low.toLocaleString("ja-JP",   { maximumFractionDigits: 0 })}` : "—", color: "text-red-400" },
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
                    <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full"
                        style={{ width: `${Math.max(0, Math.min(100, ((quote.price - quote.fiftyTwoWeekLow) / (quote.fiftyTwoWeekHigh - quote.fiftyTwoWeekLow)) * 100))}%` }}
                      />
                    </div>
                  )}
                  <span className="font-mono text-green-400">
                    高値 {quote.fiftyTwoWeekHigh != null ? `¥${quote.fiftyTwoWeekHigh.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}` : "—"}
                  </span>
                </div>
              )}
            </div>
          ) : null}
        </section>
      )}

      {/* ── 最新決算 ──────────────────────────────────────── */}
      {!loading && earnings.length > 0 && (
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
            <p className="text-xs text-gray-600 px-4 py-2 bg-gray-900/50 border-t border-gray-800">出典: Yahoo Finance</p>
          </div>
        </section>
      )}
    </div>
  );
}
