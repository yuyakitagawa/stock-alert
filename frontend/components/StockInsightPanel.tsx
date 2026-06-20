"use client";
import { useState, useEffect } from "react";
import type { Insight } from "@/lib/types";

interface Props {
  code: string;
  name: string;
  sector?: string | null;
  pbr?: number | null;
  per?: number | null;
  net?: number | null;
}

export default function StockInsightPanel({ code, name, sector, pbr, per, net }: Props) {
  const [insight, setInsight] = useState<Insight | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const cacheKey = `insight:${code}`;
      try {
        const cached = localStorage.getItem(cacheKey);
        if (cached) {
          const { data, ts } = JSON.parse(cached);
          if (Date.now() - ts < 7 * 86400 * 1000) {
            setInsight(data);
            setLoading(false);
            return;
          }
        }
      } catch { /* ignore */ }

      const params = new URLSearchParams({
        name,
        sector: sector ?? "",
        pbr: pbr != null ? String(pbr) : "",
        per: per != null ? String(per) : "",
        net: net != null ? String(net) : "",
      });
      const res = await fetch(`/api/stock/${code}/insight?${params}`);
      const data = res.ok ? (await res.json()).insight : null;
      if (cancelled) return;
      setInsight(data);
      setLoading(false);
      if (data) {
        try { localStorage.setItem(cacheKey, JSON.stringify({ data, ts: Date.now() })); } catch { /* ignore */ }
      }
    }
    load();
    return () => { cancelled = true; };
  }, [code, name, sector, pbr, per, net]);

  if (loading) {
    return (
      <section className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
        <div className="h-3 bg-gray-800 rounded w-40 animate-pulse" />
        <div className="h-3 bg-gray-800 rounded w-full animate-pulse" />
        <div className="h-3 bg-gray-800 rounded w-5/6 animate-pulse" />
      </section>
    );
  }
  if (!insight) return null;

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide">企業インサイト</h2>
        <span className="text-[11px] text-amber-500/80 bg-amber-950/30 border border-amber-900/40 px-2 py-0.5 rounded-full">
          🤖 AI生成・参考
        </span>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-2">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wide">事業概要</h3>
          <p className="text-sm text-gray-300 leading-relaxed">{insight.business}</p>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-2">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wide">主要取引先・顧客</h3>
          <p className="text-sm text-gray-300 leading-relaxed">{insight.customers}</p>
          <p className="text-[11px] text-gray-600">※社名はAI推定を含むため要確認（一次情報は有価証券報告書）</p>
        </div>

        <div className="bg-blue-950/15 border border-blue-900/30 rounded-xl p-5 space-y-2">
          <h3 className="text-xs font-bold text-blue-400 uppercase tracking-wide">カタリスト評価</h3>
          <p className="text-sm text-gray-300 leading-relaxed">{insight.catalyst}</p>
        </div>

        {insight.risks?.length > 0 && (
          <div className="bg-red-950/15 border border-red-900/30 rounded-xl p-5 space-y-2">
            <h3 className="text-xs font-bold text-red-400 uppercase tracking-wide">リスク</h3>
            <ul className="space-y-1.5">
              {insight.risks.map((r, i) => (
                <li key={i} className="flex gap-2 text-sm text-gray-300">
                  <span className="text-red-400 mt-0.5 shrink-0">!</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <p className="text-[11px] text-gray-600">
        本セクションはAIが公開情報を基に生成した参考情報で、正確性・最新性を保証しません。投資判断はご自身の責任で。
      </p>
    </section>
  );
}
